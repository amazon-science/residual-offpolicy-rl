# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Utilities for loading and saving model checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
import wandb

from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


def _download_from_wandb(checkpoint_spec: str) -> tuple[Path, dict]:
    """Download checkpoint from W&B.

    Args:
        checkpoint_spec: W&B specification in format:
            - "entity/project/runs/run_id/files/path/to/checkpoint.pt"
            - "run_id" (uses current project/entity from wandb.run)

    Returns:
        (checkpoint_path, wandb_config): Path to downloaded file and W&B run config

    Raises:
        ImportError: If wandb is not available
        ValueError: If checkpoint_spec format is invalid
    """
    # Parse the checkpoint specification
    # Extract entity/project/runs/run_id from the full path
    parts = checkpoint_spec.split("/")

    entity = parts[0]
    project = parts[1]
    run_id = parts[3]

    # Extract the file path within the run (everything after "files/")
    files_idx = parts.index("files")
    file_path = "/".join(parts[files_idx + 1 :])

    # Create API instance
    api = wandb.Api()

    # Construct run path
    run_path = f"{entity}/{project}/{run_id}"

    print(f"Downloading checkpoint from W&B run: {run_path}")
    print(f"File path: {file_path}")

    run = api.run(run_path)

    # Get the W&B config
    wandb_config = dict(run.config)

    # Create temporary directory for download
    temp_dir = Path(tempfile.mkdtemp(prefix="wandb_checkpoint_"))

    # Download the file
    downloaded_file = run.file(file_path).download(root=str(temp_dir), replace=True)
    checkpoint_path = Path(downloaded_file.name)

    print(f"✅ Downloaded checkpoint to: {checkpoint_path}")
    return checkpoint_path, wandb_config


def save_checkpoint(
    agent: QAgent,
    checkpoint_path: str | Path,
    global_step: int,
    config: Any = None,
    success_rate: float | None = None,
    **extra_data: Any,
) -> None:
    """Save a QAgent checkpoint.

    Args:
        agent: The QAgent to save
        checkpoint_path: Path where to save the checkpoint
        global_step: Current training step
        config: Training configuration (optional)
        success_rate: Success rate when checkpoint was saved (optional)
        **extra_data: Additional data to include in checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "agent_state_dict": agent.state_dict(),
        "global_step": global_step,
        **extra_data,
    }

    # Save optimizer and scheduler states
    optimizer_state_dict = {
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),
    }
    scheduler_state_dict = {}
    if hasattr(agent, "actor_scheduler") and agent.actor_scheduler is not None:
        scheduler_state_dict["actor_scheduler"] = agent.actor_scheduler.state_dict()
    if hasattr(agent, "critic_scheduler") and agent.critic_scheduler is not None:
        scheduler_state_dict["critic_scheduler"] = agent.critic_scheduler.state_dict()
    if hasattr(agent, "encoder_scheduler") and agent.encoder_scheduler is not None:
        scheduler_state_dict["encoder_scheduler"] = agent.encoder_scheduler.state_dict()

    checkpoint_data["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict:  # Only add if there are schedulers
        checkpoint_data["scheduler_state_dict"] = scheduler_state_dict

    if config is not None:
        # Convert OmegaConf to plain dict to avoid PyTorch 2.6+ loading issues
        try:
            from omegaconf import OmegaConf

            if hasattr(config, "_metadata"):  # Check if it's an OmegaConf object
                checkpoint_data["config"] = OmegaConf.to_container(config, resolve=True)
            else:
                checkpoint_data["config"] = config
        except ImportError:
            checkpoint_data["config"] = config
    if success_rate is not None:
        checkpoint_data["success_rate"] = success_rate

    torch.save(checkpoint_data, checkpoint_path)
    print(f"💾 Saved checkpoint to: {checkpoint_path}")
