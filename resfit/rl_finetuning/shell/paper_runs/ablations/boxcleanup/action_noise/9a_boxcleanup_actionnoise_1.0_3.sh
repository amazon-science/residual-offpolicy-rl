#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.random_action_noise_scale=1.0 \
    algo.use_base_policy_for_warmup=false \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=actionnoise_1.0-3 \
    wandb.notes=paper_runs/ablations/boxcleanup/action_noise/9a_boxcleanup_actionnoise_1.0_3 \
    wandb.group=actionnoise_1.0 \
    seed=3 \
    debug=false
