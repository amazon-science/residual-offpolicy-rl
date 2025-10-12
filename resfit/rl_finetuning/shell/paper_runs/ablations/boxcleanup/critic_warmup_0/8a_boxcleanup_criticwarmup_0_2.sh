#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    agent.actor.use_layer_norm=false \
    agent.critic.use_layer_norm=false \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    algo.critic_warmup_steps=0 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=criticwarmup_0-2 \
    wandb.notes=paper_runs/ablations/boxcleanup/8a_box_cleanup_criticwarmup_0_2 \
    wandb.group=criticwarmup_0 \
    seed=2 \
    debug=false
