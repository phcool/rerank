set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# export VLLM_ATTENTION_BACKEND=XFORMERS

export QRELS_FILE_PATH=/data/coding/Rearank/data/combined_qrels.txt

cd /data/coding/Rearank
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/coding/rerank/data/rearank_12k/rearank_12k__default__train__elimination_sort.parquet \
    data.val_files=/data/coding/rerank/data/rearank_12k/rearank_12k__default__train__elimination_sort.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=false \
    trainer.logger=['console'] \
    trainer.project_name='deeprerank' \
    trainer.experiment_name='rearank_7b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    custom_reward_function.path=/data/coding/rerank/compare_rounds_reward.py \
    custom_reward_function.name=compute_score