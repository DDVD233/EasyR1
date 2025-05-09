set -x

source /home/dvdai/miniconda3/etc/profile.d/conda.sh
conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging_nobox.yaml \
    algorithm.adv_estimator=grpo \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=1
