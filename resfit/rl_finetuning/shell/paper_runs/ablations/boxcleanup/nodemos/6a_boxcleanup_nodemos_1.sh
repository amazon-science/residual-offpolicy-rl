#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.offline_fraction=0.0 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=nodemos-1 \
    wandb.notes=paper_runs/ablations/boxcleanup/6a_box_cleanup_nodemos_1 \
    wandb.group=nodemos \
    seed=1 \
    debug=false
 