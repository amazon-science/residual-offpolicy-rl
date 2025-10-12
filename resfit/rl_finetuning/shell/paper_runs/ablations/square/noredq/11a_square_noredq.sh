#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    algo.prefetch_batches=4 \
    algo.n_step=1 \
    agent.critic.num_q=2 \
    agent.critic.policy_gradient_type=q1 \
    wandb.project=robomimic-square-ablations \
    wandb.name=noredq \
    wandb.notes=paper_runs/ablations/square/noredq/11a_square_noredq \
    wandb.group=noredq \
    debug=false