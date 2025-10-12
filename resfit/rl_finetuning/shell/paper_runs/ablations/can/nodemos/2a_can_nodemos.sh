#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    algo.offline_fraction=0.0 \
    wandb.project=robomimic-can-ablations \
    wandb.name=nodemos \
    wandb.notes=paper_runs/ablations/can/nodemos/2a_can_nodemos \
    wandb.group=nodemos \
    debug=false
