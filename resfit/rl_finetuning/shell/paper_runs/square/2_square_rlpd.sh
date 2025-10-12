#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_rlpd_dexmg \
    --config-name=rlpd_square_config \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-square-final \
    wandb.name=rlpd \
    wandb.notes=paper_runs/square/2_square_rlpd \
    wandb.group=rlpd \
    debug=false
