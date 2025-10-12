#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.sampling_strategy=prioritized_replay \
    wandb.project=robomimic-square-ablations \
    wandb.name=per \
    wandb.notes=paper_runs/ablations/square/per/7a_square_per \
    wandb.group=per \
    debug=false
