#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    agent.actor.use_layer_norm=false \
    agent.critic.use_layer_norm=false \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    algo.critic_warmup_steps=25_000 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=criticwarmup_25k-3 \
    wandb.notes=paper_runs/ablations/boxcleanup/8b_box_cleanup_criticwarmup_25k_3 \
    wandb.group=criticwarmup_25k \
    seed=3 \
    debug=false
