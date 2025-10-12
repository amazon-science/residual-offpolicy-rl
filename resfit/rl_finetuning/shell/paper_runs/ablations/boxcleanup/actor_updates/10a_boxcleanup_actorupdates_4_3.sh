#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.actor_updates_per_iteration=4 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=actorupdates_4-3 \
    wandb.notes=paper_runs/ablations/boxcleanup/actor_updates/10a_boxcleanup_actorupdates_4_3 \
    wandb.group=actorupdates_4 \
    seed=3 \
    debug=false
