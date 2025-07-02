set -x

/home/dvdai/miniconda3/bin/conda activate easyr1

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/scratch/high_modality/unified_train_upsampled.jsonl \
    data.val_files=/scratch/high_modality/unified_valid_mini.jsonl \
    data.image_dir=/scratch/high_modality \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/home/abthebee/EasyR1/verl/models/transformers/time_series_qwen2_5_vl \
    trainer.n_gpus_per_node=1 \
    worker.actor.optim.lr=5e-6 \
    trainer.experiment_name=drpo_vanilla_unified
