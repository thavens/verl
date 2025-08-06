#!/usr/bin/env bash
output_dir=${1:-/storage_fast/models/michael_lavery}

set -xeuo pipefail

project_name='verl_grpo_pir'
experiment_name='q2.5_3b_dxml_test'

# ppo mini batch size 128 -> 512 global batch size per gradient step
# pop mini batch size 64 -> 256 global batch size per gradient step
uv run python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./pir_data_xml_grpo.parquet \
    data.val_files=./pir_data_xml_grpo.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/scratch/public_models/huggingface/Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.rollout.load_format="safetensors" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    custom_reward_function.path=pir_reward.py \
    custom_reward_function.name=no_think \
    trainer.default_local_dir="${output_dir}/${project_name}/${experiment_name}"