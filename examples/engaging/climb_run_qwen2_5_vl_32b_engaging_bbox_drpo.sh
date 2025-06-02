set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging_bf16.yaml \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-32B-Instruct \
    trainer.n_gpus_per_node=4 \
    worker.rollout.tensor_parallel_size=2 \
    worker.actor.optim.lr=5e-6 \
    trainer.load_checkpoint_path=/scratch/dvdai/checkpoints/easy_r1/drpo_32b/global_step_140 \
    trainer.val_before_train=True \
    trainer.experiment_name=drpo_32b