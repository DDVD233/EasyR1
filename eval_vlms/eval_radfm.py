#!/usr/bin/env python3
"""
Script for evaluating RadFM model through local inference.
"""

import argparse
import json
import logging
import os
import time
import traceback
import uuid
from typing import Dict, List

import torch
from dataset import SimpleDataset

# Import evaluation metrics and dataset from the provided code
from evaluation import compute_metrics_by_data_source

# Import RadFM model components
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import LlamaTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("radfm_eval.log"), logging.StreamHandler()],
)
logger = logging.getLogger("RadFM_Local_Eval")


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    """
    Initialize the tokenizer with special tokens for image handling

    Args:
        tokenizer_path: Path to the base tokenizer
        max_img_size: Maximum number of images supported in a prompt
        image_num: Number of token embeddings per image

    Returns:
        Tuple of (tokenizer, image_padding_tokens)
    """
    image_padding_tokens = []
    # Load the base tokenizer from the provided path
    text_tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_path,
    )
    # Define initial special tokens for image markup
    special_token = {"additional_special_tokens": ["<image>", "</image>"]}

    # Generate unique tokens for each image position and patch
    for i in range(max_img_size):
        image_padding_token = ""

        for j in range(image_num):
            image_token = f"<image{i * image_num + j}>"
            image_padding_token = image_padding_token + image_token
            special_token["additional_special_tokens"].append(image_token)

        # Store the concatenated tokens for each image
        image_padding_tokens.append(image_padding_token)

    # Add all special tokens to the tokenizer
    text_tokenizer.add_special_tokens(special_token)

    # Configure standard special tokens for LLaMA models
    text_tokenizer.pad_token_id = 0
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


def prepare_model_input(prompt: str, image_paths: List[str], tokenizer, image_padding_tokens):
    """
    Prepare input for RadFM model.

    Args:
        prompt: The text prompt
        image_paths: List of image file paths
        tokenizer: Tokenizer for text processing
        image_padding_tokens: Special tokens for image placeholders

    Returns:
        Tuple of (tokenized_text, processed_images_tensor)
    """
    # Define image transformation pipeline
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                [512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )

    # Format for RadFM requires images with positions
    image_list = []
    for idx, img_path in enumerate(image_paths):
        # Place images at the beginning of the text
        image_list.append(
            {
                "img_path": img_path,
                "position": 0,  # Insert at the beginning
            }
        )

    # Process text with image placeholders and convert images
    images = []
    new_questions = [_ for _ in prompt]  # Convert prompt string to list of characters
    padding_index = 0

    # Process each image in the list
    for img in image_list:
        img_path = img["img_path"]
        position = img["position"]

        try:
            # Load and transform the image
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0).unsqueeze(-1)  # Add batch and depth dimensions

            # Resize the image to target dimensions
            target_H = 512
            target_W = 512
            target_D = 4
            images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))

            # Insert image placeholder token at the specified position
            new_questions[position] = (
                "<image>" + image_padding_tokens[padding_index] + "</image>" + new_questions[position]
            )
            padding_index += 1
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            continue

    # If no images were processed successfully, return None
    if not images:
        return None, None

    # Stack all images into a batch
    vision_x = torch.cat(images, dim=1).unsqueeze(0)

    # Join the character list back into a string and tokenize
    text = "".join(new_questions)
    lang_x = tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")["input_ids"]

    return lang_x, vision_x


def load_model(model_path, checkpoint_path, device="cuda"):
    """
    Load RadFM model from checkpoint.

    Args:
        model_path: Path to the language model files
        checkpoint_path: Path to the model weights
        device: Device to run inference on

    Returns:
        Loaded RadFM model
    """
    try:
        # Initialize the multimodal model
        model = MultiLLaMAForCausalLM(
            lang_model_path=model_path,
        )

        # Load pretrained model weights
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

        # Move model to selected device and set to evaluation mode
        model = model.to(device)
        model.eval()

        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def load_results_file(results_file: str) -> Dict:
    """Load existing results from file if it exists."""
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Error loading results file {results_file}, creating new file")
            return {"results": []}
    return {"results": []}


def save_results(results: Dict, results_file: str):
    """Save results to file."""
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


def evaluate_dataset(
    dataset_path: str,
    model_path: str,
    checkpoint_path: str,
    output_dir: str = "results",
    device: str = "cuda",
    batch_size: int = 1,  # Maintaining parameter for compatibility
):
    """
    Evaluate RadFM on a dataset through local inference.

    Args:
        dataset_path: Path to the dataset file
        model_path: Path to the language model files
        checkpoint_path: Path to the model weights
        output_dir: Directory to save results
        device: Device to run inference on
        batch_size: Placeholder for API compatibility (not used)
    """
    # Define system prompt
    system_prompt = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
Before analyzing medical images, you must identify and outline all objects of interest that are relevant to diagnosis in json format, with list od bounding box in a key called "bbox_2d" with the format [x1, y1, x2, y2]. The json should be wrapped in ```json ... ``` tags. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."""

    # Prepare output directory and results file
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(model_path)
    results_file = os.path.join(output_dir, f"{model_name}_results.json")

    # Load dataset
    dataset = SimpleDataset(data_path=dataset_path)
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Load existing results if any
    results_data = load_results_file(results_file)
    completed_ids = {item["id"] for item in results_data["results"]}
    logger.info(f"Found {len(completed_ids)} already completed examples")

    # Initialize tokenizer and model
    tokenizer, image_padding_tokens = get_tokenizer(model_path)
    model = load_model(model_path, checkpoint_path, device)

    if model is None:
        logger.error("Failed to load model, evaluation aborted")
        return

    # Evaluation loop
    all_responses = []
    all_ground_truths = []
    all_datasets = []
    all_data_sources = []

    for idx in tqdm(range(len(dataset))):
        # Generate a deterministic ID based on dataset path and example index
        example_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_path}_{idx}"))

        # Skip if already processed
        if example_id in completed_ids:
            logger.debug(f"Skipping already processed example {example_id}")

            # Add to our collection lists for metric computation
            for item in results_data["results"]:
                if item["id"] == example_id:
                    all_responses.append(item["model_response"]["content"])
                    all_ground_truths.append(item["ground_truth"])
                    all_datasets.append(item["dataset"])
                    all_data_sources.append(item["data_source"])
                    break

            continue

        try:
            # Get example data
            example = dataset[idx]
            prompt = example[dataset.prompt_key]
            ground_truth = example["ground_truth"]
            image_paths = example.get("full_image_paths", [])
            data_source = example.get("data_source", "unknown")
            dataset_name = example.get("dataset", "unknown")

            # Prepare input for the model
            lang_x, vision_x = prepare_model_input(
                prompt=prompt, image_paths=image_paths, tokenizer=tokenizer, image_padding_tokens=image_padding_tokens
            )

            # Skip if input preparation failed
            if lang_x is None or vision_x is None:
                logger.warning(f"Skipping example {example_id} due to input preparation failure")
                continue

            # Move inputs to device
            lang_x = lang_x.to(device)
            vision_x = vision_x.to(device)

            # Run inference with the model
            start_time = time.time()
            with torch.no_grad():
                try:
                    # Generate text response
                    generation = model.generate(lang_x, vision_x)

                    # Decode the generated token IDs to text
                    response_content = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
                except Exception as e:
                    logger.error(f"Inference error for example {example_id}: {e}")
                    response_content = ""

            elapsed_time = time.time() - start_time

            # Format model response like API response for consistency
            model_response = {"content": response_content}

            # Store result
            result_item = {
                "id": example_id,
                "data_source": data_source,
                "dataset": dataset_name,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "model_response": model_response,
                "inference_time_seconds": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add to results
            results_data["results"].append(result_item)

            # Add to our collection lists for metric computation
            all_responses.append(model_response["content"])
            all_ground_truths.append(ground_truth)
            all_datasets.append(dataset_name)
            all_data_sources.append(data_source)

            # Save after each example in case of interruption
            save_results(results_data, results_file)

        except Exception as e:
            logger.error(f"Error processing example {example_id}: {e}")
            logger.error(traceback.format_exc())
            continue

    # Compute and save metrics once all examples are processed
    if all_responses:
        try:
            metrics = compute_metrics_by_data_source(
                predictions=all_responses,
                ground_truths=all_ground_truths,
                data_sources=all_data_sources,
                datasets=all_datasets,
            )

            # Save metrics
            metrics_file = os.path.join(output_dir, f"{model_name}_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            # Log overall metrics
            logger.info("Evaluation completed. Overall metrics:")
            for metric_name, value in metrics.items():
                if metric_name.startswith("val/"):
                    logger.info(f"  {metric_name}: {value:.4f}")

        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
    else:
        logger.warning("No responses collected, skipping metrics computation")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RadFM on a dataset through local inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset JSON file",
        default="/mnt/8T/high_modality/geom_valid_mini_images_full.jsonl",
    )
    parser.add_argument("--model_path", type=str, help="Path to the language model files", default="./Language_files")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model weights (pytorch_model.bin)",
        default="pytorch_model.bin",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu)")

    args = parser.parse_args()

    evaluate_dataset(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
