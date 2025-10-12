#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_pouring_config \
    algo.prefetch_batches=4 \
    wandb.project=dexmg-pouring-final \
    wandb.name=residual-rl \
    wandb.notes=paper_runs/pouring/1_pouring_residual_rl \
    wandb.group=residual-rl \
    debug=false
