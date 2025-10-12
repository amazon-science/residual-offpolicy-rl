#!/usr/bin/env bash

# Launch ACT BC training for single-arm Transport task from MimicGen
# This script uses the transport dataset at ankile/mimicgen-d0-transport
# Based on the Threading task parameters but adapted for Transport

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-mh-transport-image \
    --policy act \
    --batch_size 256 \
    --wandb_project robomimic-transport-bc \
    --wandb_enable \
    --eval_env Transport \
    --rollout_freq 5000 \
    --steps 100000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 10000
