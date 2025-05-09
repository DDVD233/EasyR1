set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/scratch/high_modality/unified_train.json \
    data.val_files=/scratch/high_modality/unified_valid.json \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=drpo_vanilla_unified