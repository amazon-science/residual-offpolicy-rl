#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    algo.n_step=5 \
    wandb.project=robomimic-can-ablations \
    wandb.name=nstep5 \
    wandb.notes=paper_runs/ablations/can/nstep/4b_can_nstep5 \
    wandb.group=nstep5 \
    debug=false
