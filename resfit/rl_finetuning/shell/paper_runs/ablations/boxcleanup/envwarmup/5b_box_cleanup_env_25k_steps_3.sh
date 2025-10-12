#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    algo.learning_starts=25_000 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=env-warmup-25k-steps-3 \
    wandb.notes=paper_runs/ablations/boxcleanup/5b_box_cleanup_env_25k_steps_3 \
    wandb.group=env-warmup-25k-steps \
    seed=3 \
    debug=false
 