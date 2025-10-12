#!/usr/bin/env bash

# Launch ACT BC training for Lift task from Robomimic

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-mh-lift-image \
    --policy act \
    --batch_size 256 \
    --wandb_project robomimic-lift-bc \
    --wandb_enable \
    --eval_env Lift \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 1000
