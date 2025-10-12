#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.num_updates_per_iteration=1 \
    wandb.project=robomimic-square-ablations \
    wandb.name=utd1 \
    wandb.notes=paper_runs/ablations/square/utd/6c_square_utd1 \
    wandb.group=utd1 \
    debug=false