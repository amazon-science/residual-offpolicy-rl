#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    agent.actor.use_layer_norm=false \
    agent.critic.use_layer_norm=false \
    wandb.project=robomimic-square-ablations \
    wandb.name=nolayernorm \
    wandb.notes=paper_runs/ablations/square/nolayernorm/3a_square_nolayernorm \
    wandb.group=nolayernorm \
    debug=false
