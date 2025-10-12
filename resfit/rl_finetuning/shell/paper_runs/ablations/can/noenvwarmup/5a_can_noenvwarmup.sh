#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.learning_starts=100 \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-can-ablations \
    wandb.name=noenvwarmup \
    wandb.notes=paper_runs/ablations/can/noenvwarmup/5a_can_noenvwarmup \
    wandb.group=noenvwarmup \
    debug=false
