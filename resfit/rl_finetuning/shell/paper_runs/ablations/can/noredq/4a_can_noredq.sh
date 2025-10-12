#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    agent.critic.num_q=2 \
    agent.critic.policy_gradient_type=q1 \
    wandb.project=robomimic-can-ablations \
    wandb.name=noredq \
    wandb.notes=paper_runs/ablations/can/noredq/4a_can_noredq \
    wandb.group=noredq \
    debug=false
 