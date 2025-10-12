#!/usr/bin/env bash

# Launch ACT BC training for Square task from Robomimic
# This matches the example command provided by the user

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-mh-square-image \
    --policy act \
    --batch_size 256 \
    --wandb_project robomimic-square-bc \
    --wandb_enable \
    --eval_env Square \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 1000
