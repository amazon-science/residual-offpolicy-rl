#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.num_updates_per_iteration=2 \
    wandb.project=robomimic-square-ablations \
    wandb.name=utd2 \
    wandb.notes=paper_runs/ablations/square/utd/6d_square_utd2 \
    wandb.group=utd2 \
    debug=false