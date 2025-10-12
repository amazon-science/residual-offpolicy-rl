#!/usr/bin/env bash

# Launch ACT BC training for TwoArmThreading task from DexMimicGen
# This script uses the two-arm threading dataset at ankile/dexmg-two-arm-threading

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-threading \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmg-twoarmthreading-bc \
    --wandb_enable \
    --eval_env TwoArmThreading \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.frontview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 5000
