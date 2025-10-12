#!/usr/bin/env bash

# Launch ACT BC training for two-arm DrawerCleanup task from DexMimicGen
# This script uses the drawer cleanup dataset at ankile/dexmg-two-arm-drawer-cleanup
# Based on the Transport task parameters but adapted for DrawerCleanup

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-drawer-cleanup \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmimicgen-drawercleanup-bc \
    --wandb_enable \
    --eval_env TwoArmDrawerCleanup \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 1000
