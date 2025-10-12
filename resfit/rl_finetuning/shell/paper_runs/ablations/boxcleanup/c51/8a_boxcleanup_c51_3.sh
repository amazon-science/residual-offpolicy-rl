#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    agent.actor.action_scale=0.2 \
    agent.critic.loss.type=c51 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=c51-3 \
    wandb.notes=paper_runs/ablations/boxcleanup/8a_box_cleanup_c51_3 \
    wandb.group=c51 \
    seed=3 \
    debug=false
 