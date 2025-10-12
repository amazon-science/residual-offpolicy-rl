#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_rlpd_dexmg \
    --config-name=rlpd_pouring_config \
    algo.prefetch_batches=4 \
    wandb.project=dexmg-pouring-final \
    wandb.name=rlpd \
    wandb.notes=paper_runs/pouring/2_pouring_rlpd \
    wandb.group=rlpd \
    debug=false
