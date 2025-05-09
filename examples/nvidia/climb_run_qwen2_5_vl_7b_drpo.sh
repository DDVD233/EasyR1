set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/home/jovyan/workspace/high_modality/geom_train_upsampled.jsonl \
    data.val_files=/home/jovyan/workspace/high_modality/geom_valid_mini.jsonl \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/home/jovyan/workspace/qwen25_vision_model \
    trainer.load_checkpoint_path=/home/jovyan/workspace/EasyR1/checkpoints/easy_r1/drpo_new_nvidia_custom_encoder_ups/global_step_100 \
    trainer.n_gpus_per_node=8 \
    trainer.experiment_name=drpo_new_nvidia_custom_encoder_ups