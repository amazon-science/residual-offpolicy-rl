#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.critic_warmup_steps=0 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=criticwarmup_0-1 \
    wandb.notes=paper_runs/ablations/boxcleanup/critic_warmup_0/8a_boxcleanup_criticwarmup_0_1 \
    wandb.group=criticwarmup_0 \
    seed=1 \
    debug=false
