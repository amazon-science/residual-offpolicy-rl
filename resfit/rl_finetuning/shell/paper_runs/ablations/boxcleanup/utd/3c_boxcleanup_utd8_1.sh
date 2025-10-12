#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.num_updates_per_iteration=8 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=utd8-1 \
    wandb.notes=paper_runs/ablations/boxcleanup/3c_boxcleanup_utd8_1 \
    wandb.group=utd8 \
    seed=1 \
    debug=false
 