#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.random_action_noise_scale=0.1 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=actionnoise_0.1-1 \
    wandb.notes=paper_runs/ablations/boxcleanup/action_noise/9b_boxcleanup_actionnoise_0.1_1 \
    wandb.group=actionnoise_0.1 \
    seed=1 \
    debug=false
