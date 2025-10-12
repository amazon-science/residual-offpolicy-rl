#!/usr/bin/env bash

# Launch ACT BC training for TwoArmCoffee task from DexMimicGen
# This script uses the two-arm coffee dataset at ankile/dexmg-two-arm-coffee

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-coffee \
    --policy act \
    --steps 200000 \
    --batch_size 256 \
    --wandb_project dexmg-coffee-bc \
    --eval_env TwoArmCoffee \
    --rollout_freq 5000 \
    --eval_video_key observation.images.frontview \
    --eval_render_size 224 \
    --eval_camera_size 224 \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --wandb_enable
