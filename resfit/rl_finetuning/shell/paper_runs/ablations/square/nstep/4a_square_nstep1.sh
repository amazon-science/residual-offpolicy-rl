#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.n_step=1 \
    wandb.project=robomimic-square-ablations \
    wandb.name=nstep1 \
    wandb.notes=paper_runs/ablations/square/nstep/4a_square_nstep1 \
    wandb.group=nstep1 \
    debug=false