set -x

/home/dvdai/miniconda3/bin/conda activate test

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/home/jovyan/workspace/high_modality/geom_train.jsonl \
    data.val_files=/home/jovyan/workspace/high_modality/geom_valid_mini.jsonl \
    algorithm.adv_estimator=gae \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=ppo_vanilla