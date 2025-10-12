#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    algo.n_step=1 \
    wandb.project=robomimic-can-ablations \
    wandb.name=nstep1 \
    wandb.notes=paper_runs/ablations/can/nstep/4a_can_nstep1 \
    wandb.group=nstep1 \
    debug=false
