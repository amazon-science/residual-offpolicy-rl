#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.learning_starts=100 \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-square-ablations \
    wandb.name=noenvwarmup \
    wandb.notes=paper_runs/ablations/square/noenvwarmup/5a_square_noenvwarmup \
    wandb.group=noenvwarmup \
    debug=false
