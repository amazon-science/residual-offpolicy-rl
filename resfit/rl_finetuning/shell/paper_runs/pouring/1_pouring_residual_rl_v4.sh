#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_pouring_config \
    algo.prefetch_batches=4 \
    algo.n_step=3 \
    algo.gamma=0.995 \
    algo.learning_starts=10_000 \
    algo.critic_warmup_steps=10_000 \
    algo.num_updates_per_iteration=2 \
    algo.stddev_max=0.005 \
    algo.stddev_min=0.005 \
    algo.buffer_size=300_000 \
    agent.actor.action_scale=0.05 \
    agent.actor_lr=1e-6 \
    wandb.project=dexmg-pouring-final \
    wandb.name=residual-rl-v4 \
    wandb.notes=paper_runs/pouring/1_pouring_residual_rl_v4 \
    wandb.group=residual-rl-v4 \
    debug=false
 