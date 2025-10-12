#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    agent.actor.use_layer_norm=false \
    agent.critic.use_layer_norm=false \
    wandb.project=robomimic-can-ablations \
    wandb.name=nolayernorm \
    wandb.notes=paper_runs/ablations/can/nolayernorm/3a_can_nolayernorm \
    wandb.group=nolayernorm \
    debug=false
