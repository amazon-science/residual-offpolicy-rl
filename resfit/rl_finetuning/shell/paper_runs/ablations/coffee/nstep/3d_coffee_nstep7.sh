#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_coffee_config \
    algo.prefetch_batches=4 \
    algo.n_step=7 \
    algo.gamma=0.995 \
    algo.learning_starts=10_000 \
    algo.critic_warmup_steps=10_000 \
    algo.num_updates_per_iteration=4 \
    algo.stddev_max=0.025 \
    algo.stddev_min=0.025 \
    algo.buffer_size=300_000 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=1e-6 \
    wandb.project=dexmg-coffee-ablations \
    wandb.name=nstep7 \
    wandb.notes=paper_runs/ablations/coffee/nstep/3d_coffee_nstep7 \
    wandb.group=nstep7 \
    debug=false
 