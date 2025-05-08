# from models import load_encoder

# encoder, dim = load_encoder("/home/peili/EasyR1/epoch95.pth")

# import torch
# from transformers import AutoConfig
# from verl.models.transformers.time_series_qwen2_5_vl.modeling_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLForConditionalGeneration
# config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# model = TimeSeriesQwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     config=config,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
# model.time_series_embedding.encoder = encoder

# model.save_pretrained("/home/peili/EasyR1/verl/models/transformers/time_series_qwen2_5_vl", safe_serialization=True)



# import torch
# import os

# local_rank = int(os.environ.get("LOCAL_RANK", 0))  # or use RANK or pass manually
# torch.cuda.set_device(local_rank)

# print("Done!")

# import ray
# ray.init(address="10.1.25.0:6389")

# import json
# import torch
# import os

# data_dir = "/scratch/ecg"  # or any of your dataset JSONs
# json_path = os.path.join(data_dir, "Ga_train.json")

# with open(json_path, "r") as f:
#     data = json.load(f)

# missing = 0

# for entry in data:
#     # print(entry)
#     ts_path = entry["time-series"][0]

#     full_path = os.path.join(data_dir, ts_path)

#     try :
#         torch.load(full_path)
#     except:
#         print(full_path)
#         missing += 1
    
    
    
# print(f"\n✅ Done checking {len(data)} entries")
# print(f"Missing files: {missing}")

# # JS01052.pt

# import json
# import random

# List your JSON files
import json
import random
from pathlib import Path

# # Set working directory
# data_dir = Path("/scratch/ecg")
# ecg = Path("/scratch/ecg/ts_train.json")
# vision = Path("/scratch/")
# # Collect train and valid files
# train_files = list()
# valid_files = list(data_dir.glob("/scratch/ecg/ts_train.json"))

# def merge_and_shuffle(files, output_name):
#     merged = []
#     for f in files:
#         with open(f, "r") as infile:
#             merged.extend(json.load(infile))
#     random.shuffle(merged)
#     with open(data_dir / output_name, "w") as outfile:
#         json.dump(merged, outfile, indent=2)

# # Merge and save
# merge_and_shuffle(train_files, "unified_train.json")
# merge_and_shuffle(valid_files, "unified_valid.json")


# import json

# input_path = "/scratch/ecg/ts_valid.json"   # path to your input JSON file
# output_path = "/scratch/high_modality/ts_valid.json" # path to save the modified JSON file

# with open(input_path, "r") as f:
#     data = json.load(f)

# # Handle both a list of examples or a single example
# if isinstance(data, dict):
#     data = [data]

# for entry in data:
#     if "time-series" in entry:
#         entry["time-series"] = [
#             path if path.startswith("ts/") else "ts/" + path
#             for path in entry["time-series"]
#         ]

# with open(output_path, "w") as f:
#     json.dump(data, f, indent=2)



import json

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def load_jsonl(filename):
    with open(filename) as f:
        return [json.loads(line) for line in f]

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

ts_train = load_json("/scratch/high_modality/ts_train.json")
geom_train = load_jsonl("/scratch/high_modality/geom_train.jsonl")
merged_train = ts_train + geom_train
random.shuffle(merged_train)
save_json("/scratch/high_modality/unified_train.json", merged_train)

ts_valid = load_json("/scratch/high_modality/ts_valid.json")
geom_valid = load_jsonl("/scratch/high_modality/geom_valid.jsonl")
merged_valid = ts_valid + geom_valid
random.shuffle(merged_valid)
save_json("/scratch/high_modality/unified_valid.json", merged_valid)

# import torch
# import torch.distributed.tensor

# checkpoint = torch.load('/home/peili/EasyR1/checkpoints/easy_r1/unified_ts_freeze/global_step_50/actor/model_world_size_4_rank_0.pt', weights_only=False)
# print(checkpoint.keys())
# print(checkpoint['state_dict'].keys())

# from vllm.inputs import INPUT_REGISTRY
# from verl.models.transformers.time_series_qwen2_5_vl.processing_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLProcessor
# INPUT_REGISTRY.register_input_processor(TimeSeriesQwen2_5_VLProcessor)