#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.num_updates_per_iteration=8 \
    wandb.project=robomimic-square-ablations \
    wandb.name=utd8 \
    wandb.notes=paper_runs/ablations/square/utd/6e_square_utd8 \
    wandb.group=utd8 \
    debug=false