#!/usr/bin/env bash
set -euo pipefail

./resfit/lerobot/setup_lerobot.sh
./resfit/dexmg/setup_dexmg.sh

pip install wandb einops psutil