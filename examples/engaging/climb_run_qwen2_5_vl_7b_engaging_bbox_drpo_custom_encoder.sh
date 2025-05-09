set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/scratch/outputs/qwen/qwen25_vision_model \
    trainer.n_gpus_per_node=4 \
    trainer.load_checkpoint_path=/nfs/home2/dvdai/EasyR1/checkpoints/easy_r1/drpo_custom_encoder_nokl/global_step_100 \
    trainer.experiment_name=drpo_custom_encoder_nokl