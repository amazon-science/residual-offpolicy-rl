#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    agent.target_action_noise=false \
    wandb.project=robomimic-square-ablations \
    wandb.name=targetactionnoise \
    wandb.notes=paper_runs/ablations/square/targetactionnoise/10a_square_targetactionnoise \
    wandb.group=targetactionnoise \
    debug=false
