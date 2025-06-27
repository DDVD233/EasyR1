# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EasyR1 is a specialized reinforcement learning training framework for vision-language models, forked from veRL. It enables efficient training of multimodal AI models using various RL algorithms (GRPO, Reinforce++, ReMax, RLOO) with distributed computing support via Ray.

## Key Commands

### Installation and Setup
```bash
pip install -e .
```

### Training Commands
```bash
# Basic GRPO training on Geometry3K dataset
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh

# Math training with Qwen2.5-7B
bash examples/qwen2_5_7b_math_grpo.sh

# Multi-image training
bash examples/qwen2_5_vl_7b_multi_image.sh
```

### Model Management
```bash
# Merge checkpoint to Hugging Face format
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor

# Save tokenizer
python3 scripts/save_tokenizer.py
```

### Code Quality
```bash
# Format and lint code
ruff format .
ruff check . --fix
```

### Multi-node Training
```bash
# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Start Ray worker node
ray start --address=<head_node_ip>:6379

# Check Ray status
ray status
```

## Architecture Overview

### Core Framework (`/verl/`)
- **trainer/**: Main training orchestration and entry points
- **workers/**: Distributed workers (actor, critic, rollout, reward)
- **models/**: Model implementations and integrations
- **utils/**: Utilities for checkpointing, logging, datasets
- **single_controller/**: Ray-based distributed computing

### Training Scripts (`/examples/`)
- Configuration files and training scripts for different models
- Platform-specific examples (nvidia/, engaging/, baselines/)
- Reward function implementations
- Format prompt templates

### Evaluation (`/eval_vlms/`)
- Comprehensive evaluation framework for vision-language models
- Dataset handling and API-based evaluation
- Medical domain-specific evaluation components

### Key Configuration Files
- `examples/config.yaml`: Main training configuration template
- `pyproject.toml`: Python package configuration with Ruff settings
- `requirements.txt`: Core dependencies including transformers, vllm, ray

## Model Support

**Supported Models:**
- Qwen2/Qwen2.5-VL vision-language models
- Llama3/Qwen2/Qwen2.5/Qwen3 language models
- DeepSeek-R1 distill models

**RL Algorithms:**
- GRPO (Group Relative Policy Optimization)
- Reinforce++
- ReMax
- RLOO

## Training Configuration

Training is configured via YAML files with these key sections:
- `data`: Dataset configuration, prompt/answer keys, batch sizes
- `algorithm`: RL algorithm settings (KL penalty, coefficients)
- `worker`: Distributed worker configuration (actor, rollout, reward)
- `trainer`: Training parameters (epochs, logging, checkpointing)

## Hardware Requirements

Estimated VRAM requirements:
- 1.5B model: 1-2x 24GB GPUs
- 7B model: 4-8x 40GB GPUs
- 32B model: 8-16x 80GB GPUs

Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` for memory optimization.

## Development Workflow

1. **Setup**: Install dependencies with `pip install -e .`
2. **Configure**: Modify YAML configs in `examples/` for your use case
3. **Train**: Run training scripts from `examples/`
4. **Evaluate**: Use evaluation framework in `eval_vlms/`
5. **Deploy**: Merge checkpoints with `scripts/model_merger.py`

## Important Notes

- Main training entry point: `verl/trainer/main.py`
- All training scripts use the pattern: `python3 -m verl.trainer.main config=<config_file>`
- Checkpoints are saved in `checkpoints/easy_r1/` by default
- Use Ray for multi-node distributed training
- Vision-language models require special handling for image tokens and features