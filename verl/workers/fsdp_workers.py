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
"""
The main entry point to run the PPO algorithm
"""

from typing import Literal, Optional, Union, cast

import numpy as np
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
import tracemalloc
import gc
import os
from datetime import datetime
import pickle


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}
        
        # Initialize memory tracking for ALL ranks
        self.enable_memory_tracking = os.environ.get('ENABLE_MEMORY_TRACKING', '1') == '1'
        self.memory_snapshot_dir = os.environ.get('MEMORY_SNAPSHOT_DIR', '/tmp/memory_snapshots')
        self.memory_log_all_ranks = os.environ.get('MEMORY_LOG_ALL_RANKS', '1') == '1'  # Log from all ranks by default
        
        if self.enable_memory_tracking:
            # Create rank-specific directory
            self.rank_snapshot_dir = os.path.join(self.memory_snapshot_dir, f"rank_{self.rank}")
            os.makedirs(self.rank_snapshot_dir, exist_ok=True)
            tracemalloc.start()
            self.memory_snapshots = []
            self.update_count = 0
            self._print_memory_rank(f"Memory tracking enabled for rank {self.rank}. Snapshots will be saved to {self.rank_snapshot_dir}")
            
            # Initialize memory baseline
            self.baseline_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            self.baseline_cpu_memory = psutil.virtual_memory().used / (1024**3)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_ref_param_offload = self.config.ref.offload.offload_params
    
    def _print_memory_rank(self, *args, **kwargs):
        """Print memory debug info for all ranks or just rank 0."""
        if self.memory_log_all_ranks or self.rank == 0:
            print(f"[Rank {self.rank}]", *args, **kwargs)
    
    def _take_memory_snapshot(self, tag: str):
        """Take a detailed memory snapshot for debugging."""
        if not self.enable_memory_tracking:
            return
        
        # Get current memory usage
        current, peak = tracemalloc.get_traced_memory()
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        cpu_percent = psutil.virtual_memory().percent
        cpu_used_gb = psutil.virtual_memory().used / (1024**3)
        
        # Calculate deltas from baseline
        gpu_delta = gpu_allocated - self.baseline_gpu_memory
        cpu_delta = cpu_used_gb - self.baseline_cpu_memory
        
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append({
            'tag': tag,
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'role': self.role,
            'tracemalloc_current_mb': current / (1024**2),
            'tracemalloc_peak_mb': peak / (1024**2),
            'gpu_allocated_gb': gpu_allocated,
            'gpu_reserved_gb': gpu_reserved,
            'gpu_max_allocated_gb': gpu_max_allocated,
            'gpu_delta_gb': gpu_delta,
            'cpu_percent': cpu_percent,
            'cpu_used_gb': cpu_used_gb,
            'cpu_delta_gb': cpu_delta,
            'snapshot': snapshot
        })
        
        # Print summary
        self._print_memory_rank(f"\n[Memory Snapshot - {tag}]")
        self._print_memory_rank(f"  Role: {self.role}")
        self._print_memory_rank(f"  GPU: {gpu_allocated:.2f} GB allocated (Δ{gpu_delta:+.2f}), {gpu_reserved:.2f} GB reserved, max: {gpu_max_allocated:.2f} GB")
        self._print_memory_rank(f"  CPU: {cpu_percent:.1f}% ({cpu_used_gb:.2f} GB used, Δ{cpu_delta:+.2f})")
        
        # Print top Python memory allocations from tracemalloc
        if len(self.memory_snapshots) > 1:
            prev_snapshot = self.memory_snapshots[-2]['snapshot']
            top_stats = snapshot.compare_to(prev_snapshot, 'traceback')
            
            significant_changes = [stat for stat in top_stats if abs(stat.size_diff) > 10*1024*1024]  # > 10MB
            if significant_changes:
                self._print_memory_rank(f"\n  Top Python memory changes (>10MB):")
                for i, stat in enumerate(significant_changes[:5]):
                    self._print_memory_rank(f"    {stat.size_diff / (1024**2):+.2f} MB ({stat.count_diff:+d} blocks)")
                    # Show just the file and line, not full traceback
                    if stat.traceback:
                        frame = stat.traceback[-1]
                        self._print_memory_rank(f"      {frame.filename}:{frame.lineno}")
        
        # Always print tensor info if GPU memory increased significantly
        if abs(gpu_delta) > 0.1:  # More than 100MB change
            self._print_memory_rank(f"\n  GPU Tensor Analysis:")
            self._print_top_tensors()
        
        # Print largest objects by type
        if self.update_count % 5 == 0 or abs(cpu_delta) > 0.5:
            self._print_memory_rank(f"\n  Largest objects in memory:")
            self._print_largest_objects()
        
        # Synchronize across all ranks to ensure consistent logging
        if dist.is_initialized():
            dist.barrier()
    
    def _check_for_memory_leaks(self):
        """Check for potential memory leaks."""
        if not self.enable_memory_tracking or len(self.memory_snapshots) < 5:
            return
        
        # Check if memory is consistently increasing
        recent_snapshots = self.memory_snapshots[-5:]
        cpu_trend = [s['cpu_used_gb'] for s in recent_snapshots]
        gpu_trend = [s['gpu_allocated_gb'] for s in recent_snapshots]
        
        cpu_increasing = all(cpu_trend[i] <= cpu_trend[i+1] for i in range(len(cpu_trend)-1))
        gpu_increasing = all(gpu_trend[i] <= gpu_trend[i+1] for i in range(len(gpu_trend)-1))
        
        # Calculate total increase
        cpu_increase = cpu_trend[-1] - cpu_trend[0]
        gpu_increase = gpu_trend[-1] - gpu_trend[0]
        
        if (cpu_increasing and cpu_increase > 0.5) or (gpu_increasing and gpu_increase > 0.1):
            self._print_memory_rank(f"\n[WARNING] Potential memory leak detected on rank {self.rank}!")
            if cpu_increasing:
                self._print_memory_rank(f"  CPU memory increasing: {' -> '.join(f'{x:.2f}GB' for x in cpu_trend)}")
                self._print_memory_rank(f"  Total increase: {cpu_increase:.2f} GB")
            if gpu_increasing:
                self._print_memory_rank(f"  GPU memory increasing: {' -> '.join(f'{x:.2f}GB' for x in gpu_trend)}")
                self._print_memory_rank(f"  Total increase: {gpu_increase:.2f} GB")
            
            # Print tensor information
            self._print_tensor_info()
            
            # Force garbage collection
            self._print_memory_rank("  Running garbage collection...")
            collected = gc.collect()
            self._print_memory_rank(f"  Collected {collected} objects")
            torch.cuda.empty_cache()
            
            # Take another snapshot after GC
            self._take_memory_snapshot("after_gc")
    
    def _print_tensor_info(self):
        """Print information about tensors in memory."""
        try:
            # Count tensors by size
            tensor_counts = {}
            total_size = 0
            
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    size = obj.element_size() * obj.nelement()
                    shape_str = str(tuple(obj.shape))
                    key = f"{shape_str} ({obj.dtype})" 
                    if key not in tensor_counts:
                        tensor_counts[key] = {'count': 0, 'size': 0}
                    tensor_counts[key]['count'] += 1
                    tensor_counts[key]['size'] += size
                    total_size += size
            
            if tensor_counts:
                self._print_memory_rank(f"  Active GPU tensors (total: {total_size / (1024**3):.2f} GB):")
                # Sort by total size
                sorted_tensors = sorted(tensor_counts.items(), key=lambda x: x[1]['size'], reverse=True)
                for i, (shape, info) in enumerate(sorted_tensors[:10]):
                    size_gb = info['size'] / (1024**3)
                    self._print_memory_rank(f"    {shape}: {info['count']} tensors, {size_gb:.3f} GB total")
        except Exception as e:
            self._print_memory_rank(f"  Failed to get tensor info: {e}")
    
    def _print_top_tensors(self):
        """Print top tensors by total memory usage (count * size)."""
        try:
            tensor_groups = {}
            
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    size = obj.element_size() * obj.nelement()
                    shape = tuple(obj.shape)
                    dtype = str(obj.dtype)
                    key = (shape, dtype)
                    
                    if key not in tensor_groups:
                        tensor_groups[key] = {'count': 0, 'total_size': 0, 'single_size': size}
                    tensor_groups[key]['count'] += 1
                    tensor_groups[key]['total_size'] += size
            
            if tensor_groups:
                # Sort by count * size to find tensors with many instances
                sorted_by_count = sorted(tensor_groups.items(), 
                                       key=lambda x: x[1]['count'] * x[1]['single_size'], 
                                       reverse=True)[:5]
                
                for (shape, dtype), info in sorted_by_count:
                    total_gb = info['total_size'] / (1024**3)
                    single_mb = info['single_size'] / (1024**2)
                    self._print_memory_rank(f"    {shape} {dtype}: {info['count']} x {single_mb:.1f}MB = {total_gb:.3f}GB")
        except Exception as e:
            self._print_memory_rank(f"  Failed to get top tensors: {e}")
    
    def _print_largest_objects(self):
        """Print largest objects in memory by type."""
        try:
            import sys
            from collections import defaultdict
            
            type_sizes = defaultdict(lambda: {'count': 0, 'size': 0})
            
            for obj in gc.get_objects():
                try:
                    obj_size = sys.getsizeof(obj)
                    obj_type = type(obj).__name__
                    type_sizes[obj_type]['count'] += 1
                    type_sizes[obj_type]['size'] += obj_size
                except:
                    pass
            
            # Sort by total size
            sorted_types = sorted(type_sizes.items(), key=lambda x: x[1]['size'], reverse=True)[:10]
            
            for obj_type, info in sorted_types:
                size_mb = info['size'] / (1024**2)
                if size_mb > 1:  # Only show types using more than 1MB
                    self._print_memory_rank(f"    {obj_type}: {info['count']} objects, {size_mb:.1f} MB")
        except Exception as e:
            self._print_memory_rank(f"  Failed to get largest objects: {e}")

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        # create main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        # create ulysses device mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # validate and normalize config
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = (
            config.global_batch_size * config.ulysses_size
        ) // self.device_mesh.size()
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        if role != "ref":  # ref model's tokenizer is same as actor
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if role == "critic":
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
        else:
            auto_class = AutoModelForCausalLM

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        model = cast(PreTrainedModel, model)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if role == "ref":
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):  # transformers >= 4.52.0
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):  # transformers < 4.52.0
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if role in ["actor", "critic"]:
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                padding_free=self.config.critic.padding_free,
                role="critic",
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )

        if self._has_ref:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
            )

        if self._has_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:  # must after actor
            self._build_rollout()

        if self._has_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_fsdp_module,
            )

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        if "uid" in self._cache and not np.all(data.non_tensor_batch["uid"] == self._cache["uid"]):
            if self.enable_memory_tracking:
                self._print_memory_rank(f"  Clearing cache due to UID mismatch")
            self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            for multi_modal_data in data.non_tensor_batch["multi_modal_data"]:
                images, videos = [], []
                if "images" in multi_modal_data:
                    for image in multi_modal_data["images"]:
                        images.append(process_image(image, min_pixels, max_pixels))

                if "videos" in multi_modal_data:
                    for video in multi_modal_data["videos"]:
                        videos.append(process_video(video, min_pixels, max_pixels, video_fps))

                if len(images) != 0:
                    # it's necessary to add `dict` to properly convert batch features to dict
                    # otherwise the batch features will be converted to dict keys
                    # see https://github.com/hiyouga/EasyR1/pull/339
                    multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    multi_modal_inputs = {k: v.to(torch.cuda.current_device()) for k, v in multi_modal_inputs.items()}
                    batch_multi_modal_inputs.append(multi_modal_inputs)
                elif len(videos) != 0:
                    multi_modal_inputs = dict(
                        self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                    )
                    multi_modal_inputs = {k: v.to(torch.cuda.current_device()) for k, v in multi_modal_inputs.items()}
                    batch_multi_modal_inputs.append(multi_modal_inputs)
                else:  # text-only data
                    batch_multi_modal_inputs.append({})

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor
        
        if self.enable_memory_tracking:
            self.update_count += 1
            self._take_memory_snapshot(f"update_actor_start_{self.update_count}")

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        
        if self.enable_memory_tracking:
            self._take_memory_snapshot(f"update_actor_end_{self.update_count}")
            self._check_for_memory_leaks()
            
            # Clear any cached data that might be holding references
            if self.update_count % 5 == 0:
                self._print_memory_rank(f"\n[Clearing caches at update {self.update_count}]")
                if hasattr(self, '_cache'):
                    cache_size = len(str(self._cache))
                    self._print_memory_rank(f"  Cache size before clear: ~{cache_size} bytes")
                    self._cache.clear()
                gc.collect()
                torch.cuda.empty_cache()
                self._take_memory_snapshot(f"after_cache_clear_{self.update_count}")
        
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout
        
        if self.enable_memory_tracking:
            self._take_memory_snapshot(f"generate_sequences_start")

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        
        if self.enable_memory_tracking:
            self._take_memory_snapshot(f"generate_sequences_end")
        
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor
        
        if self.enable_memory_tracking:
            self._take_memory_snapshot(f"compute_log_probs_start")

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)
            print("Log probs computed.")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        
        if self.enable_memory_tracking:
            self._take_memory_snapshot(f"compute_log_probs_end")
        
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_ref_param_offload:
            load_fsdp_model(self.ref_fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_fsdp_module._handle.reshard(True)

        if self._use_ref_param_offload:
            offload_fsdp_model(self.ref_fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic
        
        if self.enable_memory_tracking:
            self.update_count += 1
            self._take_memory_snapshot(f"update_critic_start_{self.update_count}")

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    metric: np.array([value] if np.isscalar(value) else value) for metric, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        
        if self.enable_memory_tracking:
            self._take_memory_snapshot(f"update_critic_end_{self.update_count}")
            self._check_for_memory_leaks()
        
        return output