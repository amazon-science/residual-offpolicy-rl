#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    agent.target_action_noise=false \
    wandb.project=robomimic-can-ablations \
    wandb.name=targetactionnoise \
    wandb.notes=paper_runs/ablations/can/targetactionnoise/10a_can_targetactionnoise \
    wandb.group=targetactionnoise \
    debug=false
