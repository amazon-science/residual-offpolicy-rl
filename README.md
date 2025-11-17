# ResFiT

This repository contains the official release of code for paper "Residual Off-Policy RL for Finetuning Behavior Cloning Policies".

Website: https://residual-offpolicy-rl.github.io/

Paper: https://arxiv.org/abs/2509.19301

## Getting Started

### Environment Setup

#### 1. Create and Activate Conda Environment

Create a new conda environment with Python 3.10:

```bash
conda create -n residual python=3.10 -y
conda activate residual
```

#### 2. Install Core Dependencies

Install the RL finetuning dependencies:

```bash
./resfit/rl_finetuning/setup_rlpd_robosuite.sh
```

Install additional required packages:

```bash
pip install wandb
pip install draccus==0.10.0 torchrl==0.9.2
pip install hydra-core serial deepdiff matplotlib
```

#### 3. Logging into Hugging Face and wandb

Login to Hugging Face to access dataset and wandb for policy weights saving and loading:

```bash
hf auth login
wandb login
```

#### 4. Fix CUDA Support (if needed)

If you encounter CUDA-related issues, clean out CPU-only installs and reinstall CUDA-enabled packages:

```bash
# Remove CPU-only torchcodec
pip uninstall -y torchcodec

# Install CUDA-enabled wheel for CUDA 12.8
pip install --no-cache-dir torchcodec --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is enabled:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Launch training

### BC policy training

First we need to train the base BC policy. Taking TwoArmCoffee as an example:

```
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ankile/dexmg-two-arm-coffee \
    --policy act \
    --steps 200000 \
    --batch_size 256 \
    --wandb_project dexmg-bc \
    --eval_env TwoArmCoffee \
    --rollout_freq 5000 \
    --eval_video_key observation.images.frontview \
    --eval_render_size 224 \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --wandb_enable
```

After training finished, put the `wandb_project_name/run_id` into the corresponding task config in [residual_td3.py](./resfit/rl_finetuning/config/residual_td3.py).

### Residual RL training

Next we can train our residual RL policy:

```
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_coffee_config \
    algo.prefetch_batches=4 \
    algo.n_step=5 \
    algo.gamma=0.995 \
    algo.learning_starts=10_000 \
    algo.critic_warmup_steps=10_000 \
    algo.num_updates_per_iteration=4 \
    algo.stddev_max=0.025 \
    algo.stddev_min=0.025 \
    algo.buffer_size=300_000 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=1e-6 \
    wandb.project=dexmg-coffee \
    wandb.name=resfit \
    wandb.group=resfit \
    debug=false
```
