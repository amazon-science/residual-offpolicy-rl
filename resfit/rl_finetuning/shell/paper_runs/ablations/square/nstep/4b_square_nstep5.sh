#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.n_step=5 \
    wandb.project=robomimic-square-ablations \
    wandb.name=nstep5 \
    wandb.notes=paper_runs/ablations/square/nstep/4b_square_nstep5 \
    wandb.group=nstep5 \
    debug=false
