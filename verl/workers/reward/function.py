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

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[List[RewardInput]], List[RewardScore]]

# Legacy support for medical reward functions
MedicalRewardFunction = Callable[[str, str, torch.Tensor, torch.Tensor], RewardScore]


class FunctionRewardManager:
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_function_name = config.reward_function_name
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            # Support for medical reward functions with legacy interface
            if "medical" in self.reward_function_name:
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                segmentation_mask = data.batch["segmentation_mask"]
                bounding_box = data.batch["bbox"]
                score = self.reward_fn(response_str, ground_truth, segmentation_mask, bounding_box)
                print(f"{i}/{len(data)} Calculated medical reward score:", score)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
            else:
                # Standard reward function interface
                response_length = data.batch["response_mask"].sum(dim=-1)
                score = self.reward_fn(
                    {
                        "response": response_str,
                        "response_length": response_length[i],
                        "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    }
                )
                reward_tensor[i, response_length[i] - 1] = score["overall"]

            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        # Check if this is a medical reward function
        is_medical = "medical" in self.reward_function_name
        
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            input_dict = {
                "response": response_str,
                "response_length": response_length[i],
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            
            # Add medical-specific fields if this is a medical reward function
            if is_medical:
                if "segmentation_mask" in data.batch:
                    input_dict["segmentation_mask"] = data.batch["segmentation_mask"][i]
                if "bbox" in data.batch:
                    input_dict["bbox"] = data.batch["bbox"][i]
            
            reward_inputs.append(input_dict)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
