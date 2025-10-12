#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-square-final \
    wandb.name=residual-rl \
    wandb.notes=paper_runs/square/1_square_residual_rl \
    wandb.group=residual-rl \
    debug=false
