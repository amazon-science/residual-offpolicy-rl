#!/usr/bin/env bash

# Launch ACT BC training for single-arm Threading task from MimicGen
# This script uses the threading dataset at ankile/mimicgen-d0-threading
# Based on the Square task parameters but adapted for Threading

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/mimicgen-d0-threading \
    --policy act \
    --batch_size 256 \
    --wandb_project mimicgen-threading-bc \
    --wandb_enable \
    --eval_env Threading \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 1000
