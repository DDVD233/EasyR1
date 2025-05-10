set -x

/home/dvdai/miniconda3/bin/conda activate easyr1

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/scratch/high_modality/unified_train.json \
    data.val_files=/scratch/high_modality/unified_valid.json \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/home/peili/EasyR1/verl/models/transformers/time_series_qwen2_5_vl \
    trainer.n_gpus_per_node=2 \
    trainer.experiment_name=drpo_vanilla_unified