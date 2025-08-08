# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import logging

import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
import PIL
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from torchcodec.decoders import VideoDecoder

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('dataset_worker.log'), logging.StreamHandler()]
)
logger = logging.getLogger('RLHFDataset')


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    # Handle segmentation masks separately
    seg_masks = []
    max_height, max_width = 0, 0

    # First pass: collect all tensors and find max dimensions for segmentation masks
    for feature in features:
        for key, value in feature.items():
            if key == "segmentation_mask":
                assert isinstance(value, np.ndarray)
                if len(value.shape) == 3:
                    c, h, w = value.shape
                    # Update max dimensions
                    max_height = max(max_height, h)
                    max_width = max(max_width, w)
                elif len(value.shape) == 2:
                    h, w = value.shape
                    max_height = max(max_height, h)
                    max_width = max(max_width, w)
                seg_masks.append(value)
            elif isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                if key == "time-series":
                    continue
                non_tensors[key].append(value)

    # Second pass: pad segmentation masks to max dimensions
    padded_masks = []
    for mask in seg_masks:
        if mask is None:
            # Create zero array for missing segmentation masks
            padded_masks.append(np.zeros((1, max_height, max_width), dtype=np.float32))
        else:
            # Get current dimensions
            if len(mask.shape) == 3:  # [C, H, W]
                c, h, w = mask.shape
                # Calculate padding (bottom, right)
                pad_bottom = max_height - h
                pad_right = max_width - w
                # Pad the mask using numpy padding
                padded_mask = np.pad(mask, ((0, 0), (0, pad_bottom), (0, pad_right)),
                                     mode='constant', constant_values=0)
                padded_masks.append(padded_mask)
            elif len(mask.shape) == 2:  # [H, W]
                h, w = mask.shape
                # Calculate padding (bottom, right)
                pad_bottom = max_height - h
                pad_right = max_width - w
                # Pad the mask using numpy padding
                padded_mask = np.pad(mask, ((0, pad_bottom), (0, pad_right)),
                                     mode='constant', constant_values=0)
                padded_mask = padded_mask[np.newaxis, :, :]  # Add channel dimension
                padded_masks.append(padded_mask)
            else:
                # Handle unexpected shapes
                padded_masks.append(np.zeros((1, max_height, max_width), dtype=np.float32))

    # Add padded segmentation masks to non_tensors
    non_tensors["segmentation_mask"] = np.stack(padded_masks, axis=0)

    # Stack other tensors
    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    # Convert other non-tensors to arrays
    for key, value in non_tensors.items():
        if key != "segmentation_mask":  # We've already handled segmentation masks
            non_tensors[key] = np.array(value, dtype=object)

    # Combine tensors and non-tensors
    return {**tensors, **non_tensors}


def process_image(
        image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    try:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        image.load()  # avoid "Too many open files" errors
        if max_pixels is not None and (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if min_pixels is not None and (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except (OSError, IOError, PIL.UnidentifiedImageError) as e:
        logger.warning(f"Failed to load image: {str(e)}. Returning blank placeholder image.")
        # Return a blank RGB image as placeholder
        placeholder = Image.new('RGB', (224, 224), color='black')
        return placeholder
    except Exception as e:
        logger.error(f"Unexpected error loading image: {str(e)}. Returning blank placeholder image.")
        # Return a blank RGB image as placeholder
        placeholder = Image.new('RGB', (224, 224), color='black')
        return placeholder


def sample_video_frames_uniformly(
        video_path: str,
        num_frames: int,
        min_pixels: Optional[int],
        max_pixels: Optional[int]
) -> List[ImageObject]:
    """
    Sample frames uniformly from a video using torchcodec.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        min_pixels: Minimum pixels per frame
        max_pixels: Maximum pixels total (will be divided by num_frames for per-frame budget)

    Returns:
        List of PIL Images representing sampled frames
    """
    decoder = None

    try:
        # Adjust max_pixels to account for multiple frames
        if max_pixels is not None:
            max_pixels_per_frame = max_pixels // num_frames
        else:
            max_pixels_per_frame = None

        if min_pixels is not None:
            min_pixels_per_frame = min_pixels // num_frames
        else:
            min_pixels_per_frame = None

        # Pre-validation of video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise ValueError(f"Video file is empty: {video_path}")

        if file_size < 1024:  # Less than 1KB is suspicious
            logger.warning(f"Video file suspiciously small ({file_size} bytes): {video_path}")

        # Try to check file header to ensure it's a valid video file
        try:
            with open(video_path, 'rb') as f:
                header = f.read(12)
                if len(header) < 12:
                    raise ValueError(f"Cannot read file header: {video_path}")
        except IOError as e:
            raise ValueError(f"Cannot read video file: {video_path}") from e

        # Create VideoDecoder with error handling
        try:
            decoder = VideoDecoder(video_path)
        except RuntimeError as e:
            # Common torchcodec errors
            logger.warning(f"TorchCodec RuntimeError creating decoder for {video_path}: {str(e)}")
            raise
        except Exception as e:
            logger.warning(f"Failed to create VideoDecoder for {video_path}: {type(e).__name__}: {str(e)}")
            raise

        # Validate decoder and get metadata
        try:
            if not hasattr(decoder, 'metadata'):
                raise ValueError(f"Decoder has no metadata attribute for {video_path}")

            total_frames = decoder.metadata.num_frames

            # Validate frame count
            if total_frames <= 0:
                raise ValueError(f"Video has {total_frames} frames: {video_path}")

        except AttributeError as e:
            logger.warning(f"Failed to get metadata from video {video_path}: {str(e)}")
            raise
        except Exception as e:
            logger.warning(f"Error accessing video metadata for {video_path}: {str(e)}")
            raise

        # Sample frame indices uniformly
        if total_frames <= num_frames:
            # If we have fewer frames than requested, use all frames and repeat last
            indices = list(range(total_frames))
            while len(indices) < num_frames:
                indices.append(max(0, total_frames - 1))  # Ensure index is valid
        else:
            # Sample uniformly
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        # Validate indices
        indices = [min(max(0, idx), total_frames - 1) for idx in indices]

        # Get frames at sampled indices with error handling
        frames_tensor = None
        try:
            frames_data = decoder.get_frames_at(indices=indices)
            if frames_data is None:
                raise ValueError(f"get_frames_at returned None for {video_path}")

            frames_tensor = frames_data.data  # (T, C, H, W)

            if frames_tensor is None or frames_tensor.numel() == 0:
                raise ValueError(f"Empty frames tensor for {video_path}")

        except RuntimeError as e:
            # Common error when indices are out of bounds or decoder fails
            logger.warning(f"RuntimeError getting frames at indices {indices} for {video_path}: {str(e)}")
            raise
        except Exception as e:
            logger.warning(f"Failed to get frames for {video_path}: {type(e).__name__}: {str(e)}")
            raise

        # Validate frames tensor shape
        if len(frames_tensor.shape) != 4:
            raise ValueError(f"Unexpected frames tensor shape {frames_tensor.shape} for {video_path}")

        if frames_tensor.shape[0] != len(indices):
            logger.warning(f"Got {frames_tensor.shape[0]} frames but expected {len(indices)} for {video_path}")

        # Convert to PIL images and resize if needed
        pil_frames = []
        for i in range(min(frames_tensor.shape[0], num_frames)):
            try:
                # Extract single frame
                frame = frames_tensor[i]

                # Validate frame
                if frame.numel() == 0:
                    raise ValueError(f"Empty frame at index {i}")

                # Convert to numpy with proper error handling
                frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

                # Validate numpy array
                if frame_np.size == 0:
                    raise ValueError(f"Empty numpy array for frame {i}")

                # Handle different data ranges
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

                # Create PIL image
                pil_image = Image.fromarray(frame_np)

                # Validate PIL image
                if pil_image.size == (0, 0):
                    raise ValueError(f"Created empty PIL image for frame {i}")

                # Process image with size constraints
                processed_image = process_image(pil_image, min_pixels_per_frame, max_pixels_per_frame)
                pil_frames.append(processed_image)

            except Exception as e:
                logger.warning(f"Failed to process frame {i} from {video_path}: {type(e).__name__}: {str(e)}")
                # Add a black frame instead of failing completely
                placeholder_size = 224
                if max_pixels_per_frame is not None:
                    size_per_frame = int(np.sqrt(max_pixels_per_frame))
                    placeholder_size = min(size_per_frame, 512)
                pil_frames.append(Image.new('RGB', (placeholder_size, placeholder_size), color='black'))

        # Ensure we have the right number of frames
        while len(pil_frames) < num_frames:
            # Pad with black frames if needed
            placeholder_size = 224
            if max_pixels_per_frame is not None:
                size_per_frame = int(np.sqrt(max_pixels_per_frame))
                placeholder_size = min(size_per_frame, 512)
            pil_frames.append(Image.new('RGB', (placeholder_size, placeholder_size), color='black'))

        # Validate final output
        if len(pil_frames) != num_frames:
            logger.warning(f"Frame count mismatch: got {len(pil_frames)}, expected {num_frames}")
            pil_frames = pil_frames[:num_frames]  # Trim if too many

        return pil_frames

    except FileNotFoundError as e:
        logger.warning(f"Video file not found: {str(e)}")
    except ValueError as e:
        logger.warning(f"Video validation error: {str(e)}")
    except Exception as e:
        logger.warning(f"Unexpected error sampling video frames from {video_path}: {type(e).__name__}: {str(e)}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")

    finally:
        # Clean up decoder resources
        if decoder is not None:
            try:
                del decoder
                # Force garbage collection for large video decoders
                import gc
                gc.collect()
            except Exception as e:
                logger.debug(f"Error cleaning up decoder: {str(e)}")

    # Return black frames as final fallback
    logger.info(f"Returning {num_frames} black placeholder frames for {video_path}")
    placeholder_size = 224
    if max_pixels is not None:
        # Estimate a reasonable size based on max_pixels
        size_per_frame = int(np.sqrt(max_pixels // num_frames))
        placeholder_size = min(size_per_frame, 512)

    return [Image.new('RGB', (placeholder_size, placeholder_size), color='black') for _ in range(num_frames)]


def resize_bbox(bbox, original_width, original_height, new_width, new_height):
    """
    Resize bounding box coordinates based on image resizing ratio.

    Args:
        bbox (list): Original bounding box in format [x_min, y_min, x_max, y_max]
        original_width (int): Width of the original image
        original_height (int): Height of the original image
        new_width (int): Width of the resized image
        new_height (int): Height of the resized image

    Returns:
        list: Resized bounding box coordinates
    """
    # Calculate scaling factors
    width_ratio = new_width / original_width
    height_ratio = new_height / original_height

    # Apply scaling to bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Scale coordinates
    new_x_min = x_min * width_ratio
    new_y_min = y_min * height_ratio
    new_x_max = x_max * width_ratio
    new_y_max = y_max * height_ratio

    return [new_x_min, new_y_min, new_x_max, new_y_max]


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            processor: Optional[ProcessorMixin],
            prompt_key: str = "prompt",
            answer_key: str = "answer",
            image_key: str = "images",
            video_key: str = "videos",
            time_series_key: str = "time-series",
            video_fps: float = 2.0,
            max_prompt_length: int = 1024,
            truncation: str = "error",
            format_prompt: Optional[str] = None,
            min_pixels: Optional[int] = None,
            max_pixels: Optional[int] = None,
            filter_overlong_prompts: bool = True,
            filter_overlong_prompts_workers: int = 16,
            enable_time_series: bool = False,
            limit_video_frames: int = 4,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.time_series_key = time_series_key
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.enable_time_series = enable_time_series
        self.limit_video_frames = limit_video_frames

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.base_path = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
        self.image_dir = self.base_path

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        content_list = []
        if self.enable_time_series:
            content_list.append({"type": "time-series"})  # add time series token

        if prompt_str.count("<image>") > 0:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example and len(example[self.video_key]) > 0:
            assert prompt_str.count("<video>") <= 1
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example and len(example[self.image_key]) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(images=processed_images, text=[prompt],
                                          add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = copy.deepcopy(self.dataset[index])

        # Track if we converted video to frames
        video_frames = None

        # If we have videos, convert them to frames and update the prompt
        if self.video_key in example and len(example[self.video_key]) > 0:
            if "<video>" not in example[self.prompt_key]:
                logger.warning(f"Video key found in example {example.get('id', 'unknown')} but no <video> tag in prompt.")
                example[self.prompt_key] = "<video>" + example[self.prompt_key]
            videos = example.get(self.video_key, '')
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            # Convert videos to sampled frames
            try:
                video_frames = []
                for video in videos:
                    frames = sample_video_frames_uniformly(video, self.limit_video_frames, self.min_pixels, self.max_pixels)
                    video_frames.extend(frames)
            except Exception as e:
                logger.error(f"Failed to process video frames for {example.get('id', 'unknown')}: {str(e)}")
                # Fallback to a single black frame if video processing fails
                video_frames = [Image.new('RGB', (224, 224), color='black')] * self.limit_video_frames

            # Replace <video> with multiple <image> tags in the prompt
            prompt_str = example[self.prompt_key]
            example[self.prompt_key] = prompt_str.replace("<video>", "<image>" * self.limit_video_frames)

        messages = self._build_messages(example)

        # Store original image dimensions for bbox resizing
        original_dimensions = []
        processed_images = []  # Initialize for all cases
        processed_time_series = []

        ts_path = example.get('time-series', [])

        if self.enable_time_series:
            if self.time_series_key in example and example[self.time_series_key]:
                for i, time_series_item in enumerate(example[self.time_series_key]):
                    try:
                        if isinstance(time_series_item, str):
                            full_path = os.path.join(self.image_dir, time_series_item)

                            if not os.path.exists(full_path):
                                raise FileNotFoundError(f"Time series file not found: {full_path}")
                            else:
                                # Load the time series data
                                time_series = torch.load(full_path).to(torch.float32)
                                # if time_series.dtype == torch.bfloat16:
                                #     time_series = time_series.to(torch.float32)
                        else:
                            time_series = time_series_item
                        processed_time_series.append(time_series)

                    except Exception as e:
                        logger.error(traceback.format_exc())
                        time_series = torch.zeros((8, 2500), dtype=torch.float32)
                        processed_time_series.append(time_series)
            else:
                time_series = torch.zeros((8, 2500), dtype=torch.float32)
                processed_time_series.append(time_series)

        if processed_time_series:
            example[self.time_series_key] = processed_time_series

        # Check if we have video frames to use as images
        if video_frames is not None or (self.image_key in example and len(example[self.image_key]) > 0):
            if video_frames is not None:
                # Use video frames as images
                processed_images = video_frames
                for image in video_frames:
                    original_dimensions.append((image.width, image.height))
            elif self.image_key in example and len(example[self.image_key]) > 0:
                images = example.get(self.image_key, '')

                # These are image paths
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                    images = [os.path.join(self.image_dir, image) for image in images]

                for image in images:
                    # Get original dimensions before processing
                    try:
                        if isinstance(image, str):
                            img = Image.open(image)
                        else:
                            img = image
                        original_dimensions.append((img.width, img.height))
                    except Exception as e:
                        logger.warning(f"Failed to get dimensions for image: {str(e)}. Using default dimensions.")
                        original_dimensions.append((224, 224))

                    # Process the image
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            try:
                model_inputs = self.processor(text=[prompt],
                                              images=processed_images if len(processed_images) > 0 else None,
                                              time_series_data=processed_time_series if len(
                                                  processed_time_series) > 0 else None,
                                              add_special_tokens=False, return_tensors="pt")
            except Exception as e:
                print(e)
                num_images = len(processed_images)
                processed_images = [Image.new('RGB', (224, 224), color='black') for _ in range(num_images)]
                model_inputs = self.processor(text=[prompt],
                                              images=processed_images if len(processed_images) > 0 else None,
                                              time_series_data=processed_time_series if len(
                                                  processed_time_series) > 0 else None,
                                              add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            # Store the processed images for vLLM rollout worker
            example["multi_modal_data"] = {"image": processed_images} if processed_images else {}
        else:
            # If there is time series data, use processor, otherwise use tokenizer
            if processed_time_series:
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                model_inputs = self.processor([prompt], time_series_data=processed_time_series,
                                              add_special_tokens=False, return_tensors="pt")
            else:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {}

        if len(processed_images) > 0:
            image_size = processed_images[0].size
        else:
            image_size = (224, 224)

        if self.enable_time_series:
            example["multi_modal_data"][self.time_series_key] = processed_time_series

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        # Initialize default values
        target_size = (224, 224)
        if 'processed_images' in locals() and processed_images:
            target_size = processed_images[0].size

        if "segmentation_mask" in example and example["segmentation_mask"] is not None:
            try:
                # Extract dimensions from image_grid_thw (time, height, width)
                # We need the height and width for resizing
                target_height, target_width = example

                # Resize the segmentation mask to match the processed image dimensions
                resized_mask = example["segmentation_mask"].resize(
                    (target_width, target_height),
                    resample=Image.Resampling.NEAREST
                )

                mask_array = np.array(resized_mask)

                # If mask is grayscale, add channel dimension
                if len(mask_array.shape) == 2:
                    mask_array = mask_array[np.newaxis, :, :]
                # If mask is RGB but we only need one channel for segmentation
                elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                    mask_array = np.mean(mask_array, axis=2)[np.newaxis, :, :]

                # Convert to torch tensor
                # mask_tensor = torch.from_numpy(mask_array).float()
                example["segmentation_mask"] = mask_array
            except Exception as e:
                logger.error(traceback.format_exc())

        if "segmentation_mask" not in example or example["segmentation_mask"] is None:
            target_width, target_height = image_size
            # row_dict["segmentation_mask"] = torch.zeros(1, target_height, target_width, dtype=torch.float32)
            example["segmentation_mask"] = np.zeros((target_height, target_width), dtype=np.uint8)

        # Handle bounding box information
        if "bbox" in example and example["bbox"] and original_dimensions:
            try:
                # Get original dimensions of the corresponding image
                # We assume the bbox corresponds to the first image
                original_width, original_height = original_dimensions[0]
                target_width, target_height = target_size

                # Resize the bounding box
                resized_bbox = resize_bbox(
                    example["bbox"],
                    original_width,
                    original_height,
                    target_width,
                    target_height
                )

                logger.debug(f"Resized bbox from {example['bbox']} to {resized_bbox}. "
                             f"Original dimensions: {original_dimensions[0]}, "
                             f"Target dimensions: {target_width}x{target_height}")
                example["bbox"] = resized_bbox
            except Exception as e:
                logger.error(f"Error resizing bounding box: {str(e)}")
                example["bbox"] = [0, 0, 0, 0]
        else:
            # Use empty list as placeholder if not available
            example["bbox"] = [0, 0, 0, 0]

        # Make bbox tensor
        example["bbox"] = torch.tensor(example["bbox"], dtype=torch.float32)

        # Extract data_source and dataset

        # Set vision_path to a nonempty vision path
        # Or empty if both vision paths are empty
        is_timeseries = False
        vision_path = example['images'][0] if 'images' in example and len(example['images']) != 0 else None
        if vision_path is None:  # this may be video
            vision_path = example['videos'][0] if 'videos' in example and len(example['videos']) != 0 else None
        if vision_path is None:  # this may be time series only
            vision_path = example['time_series'][0] if 'time_series' in example and len(
                example['time_series']) != 0 else ''
            is_timeseries = True
        prompt_str = example[self.prompt_key]

        if 'How long will the patient stay in the hospital?' in prompt_str:
            example["data_source"] = "multimodal"
            example["dataset"] = "los_prediction"
        elif 'Will the patient survive for at least 48 hours?' in prompt_str:
            example["data_source"] = "multimodal"
            example["dataset"] = "48_ihm"
        elif len(vision_path) != 0:
            try:
                example["data_source"] = vision_path.split("/")[0]
                example["dataset"] = vision_path.split("/")[1]
            except IndexError:
                example["data_source"] = "unknown"
                example["dataset"] = "unknown"
                print(f"Failed to parse vision path: {vision_path}. The annotation is {example}. Using default values.")
        elif is_timeseries:
            example["data_source"] = "ecg"
            # dataset already set in json
        else:
            raise ValueError("No modality found.")

        example['vision_path'] = vision_path

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example[self.answer_key]

        # Clean up
        example.pop("segmentation_path", None)

        return example