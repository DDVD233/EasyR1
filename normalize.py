from pathlib import Path

import torch


data_dir = Path("/scratch/ecg/ptbxl")  # ← replace with your actual path
output_dir = data_dir  # or set to data_dir if you want to overwrite
output_dir.mkdir(exist_ok=True)


def normalize_multichannel_sample(sample):
    """
    Normalize a multichannel time series (e.g., 8-channel ECG).
    sample: Tensor of shape (channels, time_steps) = (8, 2500)
    Returns: Tensor of same shape, normalized per channel.
    """
    mean = sample.mean(dim=1, keepdim=True)  # shape: (8, 1)
    std = sample.std(dim=1, keepdim=True)  # shape: (8, 1)
    std[std == 0] = 1  # avoid divide-by-zero
    return (sample - mean) / std


for pt_file in data_dir.glob("*.pt"):
    try:
        tensor = torch.load(pt_file)
        if not isinstance(tensor, torch.Tensor):
            print(f"Skipping non-tensor file: {pt_file.name}")
            continue

        normed_tensor = normalize_multichannel_sample(tensor)

        # Save to new directory
        output_path = output_dir / pt_file.name
        torch.save(normed_tensor, output_path)
        print(f"Normalized: {pt_file.name}")

    except Exception as e:
        print(f"Error processing {pt_file.name}: {e}")
