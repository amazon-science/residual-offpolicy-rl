#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_rlpd_dexmg \
    --config-name=rlpd_can_config \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-can-final \
    wandb.name=rlpd \
    wandb.notes=paper_runs/can/2_can_rlpd \
    wandb.group=rlpd \
    debug=false
