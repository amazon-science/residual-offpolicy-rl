#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.critic_warmup_steps=25_000 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=criticwarmup_25k-2 \
    wandb.notes=paper_runs/ablations/boxcleanup/critic_warmup_25k/8b_boxcleanup_criticwarmup_25k_2 \
    wandb.group=criticwarmup_25k \
    seed=2 \
    debug=false
