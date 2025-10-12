# Running the RLPD implementation for the Robomimic/DexMimicGen tasks

## Install the required dependencies

Run this command from the root:

```bash
./resfit/rl_finetuning/setup_rlpd_robosuite.sh
```

This script will clone and install the required repos from LeRobot, Robosuite, and DexMimicGen, as well as install other dependencies we need.

This should in theory set everython up and make you ready to launch commands. The required data from Robomimic/DexMimicGen I've already converted into the LeRobot format and upload to Huggingface, so that should be no problem. However, to ensure you don't get any throttling errors when downloading the data, please run:

```bash
huggingface-cli login
```


## Run the training

The general command to run the code is

```bash
python -m resfit.rl_finetuning.scripts.train_rlpd_dexmg
```

## Other tasks

You can specify any task in the `task` field, and so far I've tested and had good results with `Lift`, `Can`, and `Square` from the Robomimic suite, and tested with the `TwoArmBoxCleanup` and `TwoArmCoffee` from the DexMimicGen suite, though the results for those tasks are still not good.

You do have to update some of the config parameters for the other tasks to work, like the `offline_data.name` to be the right dataset. All the datasets I've converted can be found here: https://huggingface.co/ankile/datasets.


## Other

If you optionally want to have replay buffer caches be persisted to a remote repo (Hugging Face Hub) so that it can be used across machines, you can set these environment variables, e.g., as such:

```bash
export HF_OFFLINE_BUFFER_REPO=ankile/rlpd-offline-buffers
export HF_ONLINE_BUFFER_REPO=ankile/rlpd-online-buffers
```


# Running the Residual implementation for the Robomimic/DexMimicGen tasks

step 1: first run BC
```
python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-box-cleanup \
    --policy act \
    --steps 100000 \
    --batch_size 128 \
    --eval_env TwoArmBoxCleanup \
    --rollout_freq 5000 \
    --eval_video_key observation.images.frontview \
    --eval_render_size 224 \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --wandb_project dexmg-box-clean-bc \
    --wandb_enable
```

step 2: then run residual RL
```
python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_box_clean_config \
    algo.n_step=3 \
    algo.num_updates_per_iteration=8 \
    algo.buffer_size=300_000 \
    algo.learning_starts=50_000 \
    algo.critic_warmup_steps=20_000 \
    algo.stddev_max=0.05 \
    algo.stddev_min=0.05 \
    algo.random_action_noise_scale=0.05 \
    offline_data.num_episodes=1000 \
    task=TwoArmBoxCleanup \
    agent.actor_lr=5e-6 \
    agent.critic_lr=1e-4 \
    agent.critic.num_q=10 \
    agent.critic_target_tau=0.005 \
    agent.actor.action_scale=0.2 \
    agent.actor.actor_last_layer_init_scale=0.0 \
    eval_interval_every_steps=5_000 \
    log_freq=100 \
    wandb.project=residual_td3 \
    wandb.mode=online \
    wandb.run_name=reproduce_box_cleanup \
    debug=false
```