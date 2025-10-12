#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.n_step=5 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=nstep5-2 \
    wandb.notes=paper_runs/ablations/boxcleanup/2b_box_cleanup_nstep5_2 \
    wandb.group=nstep5 \
    seed=2 \
    debug=false
 