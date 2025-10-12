#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_rlpd_dexmg \
    --config-name=rlpd_two_arm_cansort_config \
    algo.prefetch_batches=4 \
    algo.gamma=0.995 \
    algo.n_step=5 \
    wandb.project=dexmg-cansorting-final \
    wandb.name=rlpd \
    wandb.notes=paper_runs/cansorting/2_cansorting_rlpd \
    wandb.group=rlpd \
    debug=false
