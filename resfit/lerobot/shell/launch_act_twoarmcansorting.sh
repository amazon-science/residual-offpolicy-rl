#!/usr/bin/env bash

# Launch ACT BC training for TwoArmCanSorting task from DexMimicGen
# This script uses the two-arm can sorting dataset at ankile/dexmg-two-arm-can-sort

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-can-sort-random \
    --policy act \
    --steps 200000 \
    --batch_size 256 \
    --wandb_project dexmg-cansorting-bc \
    --eval_env TwoArmCanSortRandom \
    --rollout_freq 1_000 \
    --steps 100_000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 50 \
    --log_freq 100 \
    --save_freq 10_000 \
    --wandb_enable
