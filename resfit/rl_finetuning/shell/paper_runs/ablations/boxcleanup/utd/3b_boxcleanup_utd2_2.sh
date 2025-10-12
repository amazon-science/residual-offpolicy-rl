#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.num_updates_per_iteration=2 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=utd2-2 \
    wandb.notes=paper_runs/ablations/boxcleanup/3b_boxcleanup_utd2_2 \
    wandb.group=utd2 \
    seed=2 \
    debug=false
 