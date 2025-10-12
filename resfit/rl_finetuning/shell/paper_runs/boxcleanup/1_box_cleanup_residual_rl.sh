#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.prefetch_batches=4 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-final \
    wandb.name=residual-rl-v2 \
    wandb.notes=paper_runs/box_cleanup/1_box_cleanup_residual_rl_v2 \
    wandb.group=residual-rl-v2 \
    debug=false
 