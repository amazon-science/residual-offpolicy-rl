#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.total_timesteps=300_000 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=5e-6 \
    agent.critic.num_q=2 \
    agent.critic.policy_gradient_type=q1 \
    wandb.project=dexmg-box-clean-ablations \
    wandb.name=td3-q1-1 \
    wandb.notes=paper_runs/ablations/boxcleanup/1a_box_cleanup_td3_q1_1 \
    wandb.group=td3-q1 \
    seed=1 \
    debug=false
 