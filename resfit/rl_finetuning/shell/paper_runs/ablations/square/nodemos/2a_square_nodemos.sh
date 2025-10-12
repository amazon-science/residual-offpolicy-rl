#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.offline_fraction=0.0 \
    wandb.project=robomimic-square-ablations \
    wandb.name=nodemos \
    wandb.notes=paper_runs/ablations/square/nodemos/2a_square_nodemos \
    wandb.group=nodemos \
    debug=false
