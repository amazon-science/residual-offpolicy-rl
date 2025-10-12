#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_rlpd_dexmg \
    --config-name=rlpd_box_clean_config \
    algo.prefetch_batches=4 \
    wandb.project=dexmg-box-clean-final \
    wandb.name=rlpd \
    wandb.notes=paper_runs/box_cleanup/2_box_cleanup_rlpd \
    wandb.group=rlpd \
    debug=false
