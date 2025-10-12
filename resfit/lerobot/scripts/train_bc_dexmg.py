#!/usr/bin/env python
"""Train a LeRobot policy on a dataset hosted on the Hugging Face Hub.


Key features
============
1.  Works with any dataset created with `LeRobotDataset` and uploaded to the Hub.
2.  Supports the main LeRobot policies ("diffusion", "act", "pi0", …) with
    their default configs.
3.  Delta-timestamps are **inferred automatically** from the policy's
    *_delta_indices* properties so that the dataset delivers exactly what the
    policy expects.  The utility that does the heavy-lifting lives in
    `lerobot.common.datasets.factory.resolve_delta_timestamps`.
4.  Uses AdamW for optimisation (learning-rate & weight-decay are taken from the
    policy's config when available).
5.  Optional Weights & Biases logging -- enabled via `--wandb_enable` and the
    `--wandb_project` name.
6.  Saves checkpoints using `policy.save_pretrained(…)` so that they can later
    be re-loaded with `PreTrainedPolicy.from_pretrained(…)` **or** retrieved
    directly from W&B artefacts for evaluation.

Example
=======
python train_hf.py \
  --dataset ankile/franka-lift-dataset \
  --policy diffusion \
  --wandb_enable --wandb_project my_robot_runs
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.common.datasets.utils import cycle
from lerobot.common.utils.random_utils import set_seed
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored

import wandb
from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper, create_vectorized_env
from resfit.lerobot.policies.factory import make_policy, make_policy_config
from resfit.lerobot.policies.pretrained import PreTrainedPolicy
from resfit.lerobot.utils.load_policy import load_checkpoint, save_checkpoint

# Set multiprocessing start method for CUDA compatibility
# This must be done before any other multiprocessing operations
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set, which is fine
    pass

# -----------------------------------------------------------------------------
# Caching configuration ------------------------------------------------------
# -----------------------------------------------------------------------------
# Generic environment variable (shared across algorithms) -------------------
# ``CACHE_DIR`` specifies the root folder for **all** local caches.
# Falls back to the current directory if unset.
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()


parser = argparse.ArgumentParser(description="Offline training on a HF Hub dataset with LeRobot policies")

# Required args
parser.add_argument(
    "--dataset", type=str, required=True, help="HF Hub dataset repo-id e.g. `ankile/franka-lift-dataset`"
)
parser.add_argument(
    "--policy",
    type=str,
    default="diffusion",
    choices=[
        "diffusion",
        "act",
        "latent_act",
        "pi0",
        "pi0fast",
        "tdmpc",
        "vqbet",
    ],
    help="Which policy architecture to train",
)

# Training hyper-parameters
parser.add_argument("--steps", type=int, default=100_000, help="Total optimization steps")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--grad_clip_norm", type=float, default=10.0)
parser.add_argument("--num_workers", type=int, default=4)

# Reproducibility / device
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

# Logging & checkpoints
parser.add_argument("--output_dir", type=str, default="outputs/train_hf")
parser.add_argument("--log_freq", type=int, default=100, help="How often to print & log to W&B (in steps)")
parser.add_argument("--save_freq", type=int, default=10_000, help="How often to save checkpoints (in steps)")

# WandB
parser.add_argument("--wandb_enable", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (required when --wandb_enable)")
parser.add_argument("--wandb_entity", type=str, default=None)

# Resume
parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to a local checkpoint directory to resume from")
parser.add_argument(
    "--resume_run_id",
    type=str,
    default=None,
    help="WandB run ID to resume from (grabs the 'latest' artifact to restore trainer state).",
)

# ------------------------------------------------------------------
# Evaluation rollouts
# ------------------------------------------------------------------
parser.add_argument(
    "--rollout_freq",
    type=int,
    default=None,
    help=(
        "Frequency (in optimization steps) at which to run rollouts in a "
        "DexMimicGen environment to compute the success-rate of the current "
        "policy. Disabled when not set."
    ),
)
parser.add_argument(
    "--eval_env",
    type=str,
    default=None,
    help="DexMimicGen environment name used for evaluation rollouts.",
)
parser.add_argument("--eval_num_envs", type=int, default=5, help="Number of parallel environments for evaluation.")
parser.add_argument(
    "--eval_num_episodes", type=int, default=20, help="Total number of episodes to run during evaluation."
)
parser.add_argument(
    "--eval_camera_size", type=int, default=84, help="Camera image size for evaluation rollouts (should match dataset)."
)
parser.add_argument(
    "--eval_render_size",
    type=int,
    default=None,
    help="High-resolution camera size for video recording (if different from eval_camera_size).",
)
parser.add_argument(
    "--eval_video_key",
    type=str,
    default=None,
    help="Observation key for the camera to use for video recording (e.g., 'observation.images.frontview').",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode (uses synchronous vec envs instead of async multiprocessing for debugging).",
)

# -------------------------------------------------
# Policy configuration overrides
# -------------------------------------------------
# Allow users to pass a JSON-formatted string that will be forwarded as keyword
# arguments to the policy configuration dataclass (e.g. ACTConfig, DiffusionConfig, …).
# Example:
#   --policy_kwargs '{"dim_model": 1024, "chunk_size": 100}'
# This will result in `ACTConfig(dim_model=1024, chunk_size=100, …)`
parser.add_argument(
    "--policy_kwargs",
    type=str,
    default=None,
    help=(
        "Overrides for the policy configuration. Accepts either: "
        '1) A JSON dictionary string, e.g. \'{"dim_model": 1024, "chunk_size": 100}\', or '
        "2) A compact 'key=value' list separated by commas or spaces, e.g. "
        "   'dim_model=1024,chunk_size=100 optimizer_lr=3e-4'. "
        "All pairs are forwarded directly to make_policy_config(...)."
    ),
)

# Camera selection
parser.add_argument(
    "--policy_cameras",
    type=str,
    nargs="*",
    default=None,
    help=(
        "List of camera names to use for the policy. If not specified, all cameras from the dataset will be used. "
        "Example: --policy_cameras agentview robot0_eye_in_hand"
    ),
)

# Proprioceptive observations
parser.add_argument(
    "--disable_proprioceptive_obs",
    action="store_true",
    help=(
        "Disable proprioceptive observations (observation.state) during training. "
        "Only visual observations will be used."
    ),
)

args_cli = parser.parse_args()

# -------------------------------------------------
# Setup logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)


# Create a named logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def _annotate_frame(
    frame: np.ndarray,
    env_idx: int,
    episode_num: int,
    total_episodes: int,
    episode_step: int,
    is_success: bool,
    font=None,
) -> np.ndarray:
    """Annotate a single frame with episode information."""
    # Add text annotation to the frame
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Prepare episode status text
    episode_text = f"Env {env_idx + 1} | Episode {episode_num}/{total_episodes}"
    step_text = f"Step {episode_step}"
    status_text = "SUCCESS" if is_success else "FAIL"
    status_color = (0, 255, 0) if is_success else (255, 0, 0)

    # Add text annotations
    y_offset = 10
    draw.text((10, y_offset), episode_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), step_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), status_text, fill=status_color, font=font)

    # Convert back to numpy array
    return np.array(pil_img)


def _run_rollouts(
    *,
    policy: PreTrainedPolicy,
    env: VectorizedEnvWrapper,
    save_dir: Path,
    step: int,
    num_episodes: int,
    run_start_time: str,
):
    """Run *num_episodes* episodes with *policy* in vectorized *env* and compute success-rate.

    Captures a video, writes it to *save_dir*/`eval_step_<step>.mp4`, and returns `(success_rate, video_path)`.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    policy_was_training = policy.training
    policy.eval()

    # Get environment info
    num_parallel_envs = env.num_envs
    env_name = getattr(env, "env_name", "Unknown")

    successes = 0
    done_episodes = 0
    total_steps = 0

    start_time = time.perf_counter()

    logger.info(f"Running rollouts with environment: {env_name}")
    logger.info(f"Starting evaluation: {num_episodes} episodes using {num_parallel_envs} parallel environments")

    # Try to load a font for text annotations
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Create video writer at the beginning
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_dir = save_dir / f"eval_{env_name.lower()}" / run_start_time
    parent_dir.mkdir(parents=True, exist_ok=True)
    video_path = parent_dir / f"eval_step_{step}_{now}.mp4"

    video_writer = imageio.get_writer(video_path.as_posix(), fps=20)

    obs, _ = env.reset()
    episode_frames = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs

    while done_episodes < num_episodes:
        # Run episodes in parallel until we complete the required number
        with torch.inference_mode():
            # Convert numpy observations to PyTorch tensors for the policy
            action = policy.select_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        frames = env.render()
        for env_idx in range(num_parallel_envs):
            episode_frames[env_idx].append(frames[env_idx])
            episode_steps[env_idx] += 1

        total_steps += num_parallel_envs

        done = terminated | truncated

        if any(done):
            terminated_envs = torch.where(done)[0]
            success_envs = torch.where(reward == 1.0)[0]

            # Reset policy hidden state for the terminated envs.
            policy.reset(env_ids=terminated_envs)

            # Annotate and write frames for completed episodes immediately
            for env_idx in terminated_envs:
                env_idx_int = env_idx.item()
                is_success = env_idx in success_envs
                done_episodes += 1
                successes += int(is_success)

                # Annotate each frame in this episode and write to video
                for step_idx, frame in enumerate(episode_frames[env_idx_int]):
                    annotated_frame = _annotate_frame(
                        frame=frame,
                        env_idx=env_idx_int,
                        episode_num=done_episodes,
                        total_episodes=num_episodes,
                        episode_step=step_idx + 1,
                        is_success=is_success,
                        font=font,
                    )
                    # Write frame immediately to video
                    video_writer.append_data(annotated_frame)

                # Reset for next episode
                episode_frames[env_idx_int] = []
                episode_steps[env_idx_int] = 0

        if total_steps % 1_000 == 0:
            logger.info(
                f"Total steps: {total_steps}, done episodes: {done_episodes}, successes: {successes}, "
                f"FPS: {total_steps / (time.perf_counter() - start_time):.1f}"
            )

    video_writer.close()

    success_rate = successes / done_episodes if done_episodes > 0 else 0.0

    if policy_was_training:
        policy.train()

    # Calculate final performance metrics
    total_elapsed_time = time.perf_counter() - start_time
    final_fps = total_steps / total_elapsed_time if total_elapsed_time > 0 else 0.0
    episodes_per_sec = done_episodes / total_elapsed_time if total_elapsed_time > 0 else 0.0

    logger.info(f"Evaluation completed: {done_episodes} episodes, {successes} successes ({success_rate * 100:.1f}%)")
    logger.info(f"Performance: {total_steps} total steps in {total_elapsed_time:.1f}s")
    logger.info(f"Average FPS: {final_fps:.1f} frames/sec | Episodes/sec: {episodes_per_sec:.2f}")
    logger.info(
        f"Parallel efficiency: {num_parallel_envs} environments,"
        f" {final_fps / num_parallel_envs:.1f} frames/sec per environment"
    )
    logger.info(f"Video saved with annotated frames: {video_path}")

    return success_rate, video_path, final_fps


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(cfg: argparse.Namespace):
    # ---------------------------------------------------------------------
    # Logging / device setup
    # ---------------------------------------------------------------------
    device = torch.device(cfg.device)
    logger.info(colored(f"Using device: {device}", "green"))

    # Create run start time for organizing video outputs
    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a timestamped folder in CACHE_DIR for all outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"bc_run_{timestamp}_{Path(cfg.dataset).name}_{cfg.policy}"
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.seed is not None:
        set_seed(cfg.seed)
        logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))

    # ---------------------------------------------------------------------
    # Dataset (metadata first, then actual dataset with resolved timestamps)
    # ---------------------------------------------------------------------
    logger.info("Fetching dataset metadata from the Hub…")
    ds_meta = LeRobotDatasetMetadata(cfg.dataset)

    # ---------------------------------------------------------------------
    # Build the policy configuration, applying any CLI-specified overrides
    # ---------------------------------------------------------------------

    def _infer_type(val: str):
        """Try to cast *val* to int, float or bool if possible, else return str."""
        if val.lower() in {"true", "false"}:
            return val.lower() == "true"
        try:
            if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                return int(val)
            return float(val)
        except ValueError:
            return val  # leave as string

    if cfg.policy_kwargs is not None:
        # Initialize policy_kwargs as empty dict
        policy_kwargs = {}

        # First attempt JSON parsing ----------------------------------
        try:
            policy_kwargs = json.loads(cfg.policy_kwargs)
            if not isinstance(policy_kwargs, dict):
                raise TypeError
        except Exception:
            # Fallback: parse "key1=val1,key2=val2 ..." ----------------
            text = cfg.policy_kwargs.strip()
            # Allow comma or whitespace as separators
            tokens = re.split(r"[ ,]+", text)
            for token in filter(None, tokens):
                if "=" not in token:
                    raise ValueError(f"Could not parse --policy_kwargs token '{token}'. Expected 'key=value'.")
                k, v = token.split("=", 1)
                policy_kwargs[k] = _infer_type(v)
    else:
        policy_kwargs = {}

    # Build the policy config and set the target device.  When resuming from
    # checkpoints the device string can include an explicit index (e.g.
    # "cuda:0").  The underlying config validator only understands the bare
    # device types ("cuda", "cpu", "mps"), so we drop any optional suffix
    # before passing it on.

    policy_cfg = make_policy_config(cfg.policy, **policy_kwargs)

    # Set the chunk size to 20 (env is at 20 fps)
    policy_cfg.chunk_size = 20
    policy_cfg.n_action_steps = 20

    if isinstance(cfg.device, str):
        # e.g. "cuda:0" -> "cuda"
        policy_cfg.device = cfg.device.split(":", 1)[0]
    else:
        # Fall back to original value if somehow not a string
        policy_cfg.device = cfg.device

    # Filter dataset features based on selected cameras if specified
    if cfg.policy_cameras is not None:
        logger.info(f"Filtering dataset to use only cameras: {cfg.policy_cameras}")

        # Create a filtered features dict that only includes selected cameras
        filtered_features = {
            key: feature for key, feature in ds_meta.features.items() if not key.startswith("observation.images.")
        }

        # Only include selected camera features
        available_cameras = []
        for key, feature in ds_meta.features.items():
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                available_cameras.append(camera_name)
                if camera_name in cfg.policy_cameras:
                    filtered_features[key] = feature

        # Validate that all requested cameras exist
        missing_cameras = [cam for cam in cfg.policy_cameras if cam not in available_cameras]
        if missing_cameras:
            raise ValueError(
                f"Requested cameras not found in dataset: {missing_cameras}. Available cameras: {available_cameras}"
            )

        logger.info(f"Available cameras: {available_cameras}")
        logger.info(f"Selected cameras: {cfg.policy_cameras}")

        # Update the metadata object with filtered features
        ds_meta.info["features"] = filtered_features

    # Filter dataset features to remove proprioceptive observations if specified
    if cfg.disable_proprioceptive_obs:
        logger.info("Filtering dataset to remove proprioceptive observations (observation.state)")

        # Create a filtered features dict that excludes observation.state
        filtered_features = {key: feature for key, feature in ds_meta.features.items() if key != "observation.state"}

        # Validate that we still have some observations
        remaining_obs_keys = [key for key in filtered_features if key.startswith("observation")]
        if not remaining_obs_keys:
            raise ValueError(
                "Cannot disable proprioceptive observations: no other observation types found in dataset. "
                "Dataset must contain at least one image observation or environment state."
            )

        logger.info(f"Remaining observation keys after filtering: {remaining_obs_keys}")

        # Update the metadata object with filtered features
        ds_meta.info["features"] = filtered_features

    # Determine delta-timestamps from policy indices & dataset fps
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    logger.info("Building LeRobotDataset with inferred delta-timestamps…")

    image_transforms_config = ImageTransformsConfig(enable=True)
    image_transforms = ImageTransforms(image_transforms_config)

    dataset = LeRobotDataset(
        cfg.dataset,
        delta_timestamps=delta_timestamps,
        download_videos=True,
        image_transforms=image_transforms,
    )

    # ---------------------------------------------------------------------
    # Dataloader
    # ---------------------------------------------------------------------
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    dl_iter = cycle(dataloader)

    # ---------------------------------------------------------------------
    # Policy + optimizer
    # ---------------------------------------------------------------------
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.train()

    # Print the policy config
    print(policy_cfg)

    # Learning-rate & weight-decay fallbacks
    lr_default = getattr(policy_cfg, "optimizer_lr", 1e-4)
    wd_default = getattr(policy_cfg, "optimizer_weight_decay", 0.0)

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=lr_default, weight_decay=wd_default)

    # ---------------------------------------------------------------------
    # Optional WandB
    # ---------------------------------------------------------------------
    if cfg.wandb_enable:
        if cfg.wandb_project is None:
            raise ValueError("--wandb_project is required when --wandb_enable is set")

        wandb_run_id = cfg.resume_run_id if cfg.resume_run_id else None

        # -----------------------
        # Prepare an exhaustive config dict for W&B for improved reproducibility
        # -----------------------
        extra_cfg: dict[str, Any] = {}

        # 1) CLI / script arguments (already cover basics)
        extra_cfg.update(vars(cfg))

        # 2) Policy (actor) configuration
        try:
            extra_cfg["policy_config"] = asdict(policy_cfg) if is_dataclass(policy_cfg) else policy_cfg.__dict__
        except Exception:
            # Fallback to string repr if conversion failed (e.g. non-serialisable fields)
            extra_cfg["policy_config"] = str(policy_cfg)

        # 3) Dataset metadata (lightweight summary only to avoid huge nested structures)
        extra_cfg["dataset_meta"] = {
            "repo_id": ds_meta.repo_id,
            "fps": ds_meta.fps,
            "robot_type": ds_meta.robot_type,
            "total_episodes": ds_meta.total_episodes,
            "total_frames": ds_meta.total_frames,
            "feature_keys": list(ds_meta.features.keys()),
        }

        # 4) Delta-timestamps actually used when building the dataset
        extra_cfg["delta_timestamps"] = delta_timestamps

        # NEW: Log image transforms configuration
        extra_cfg["image_transforms"] = asdict(image_transforms_config)

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=extra_cfg,
            name=f"{cfg.policy}_{Path(cfg.dataset).name}",
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )
        logger.info(colored("W&B logging enabled", "blue"))

    # ---------------------------------------------------------------------
    # Optionally resume from checkpoint (local folder or WandB artifact)
    # ---------------------------------------------------------------------
    start_step = 0
    if cfg.resume_run_id is not None:
        logger.info(colored(f"Resuming from WandB run {cfg.resume_run_id}", "cyan"))
        api = wandb.Api()
        # Compose the artifact path: <entity>/<project>/run_<id>_latest:latest
        artifact_path = (
            f"{(cfg.wandb_entity + '/' if cfg.wandb_entity else '')}"
            f"{cfg.wandb_project}/run_{cfg.resume_run_id}_latest:latest"
        )
        artifact = api.artifact(artifact_path)
        artifact_dir = Path(artifact.download())

        start_step, policy, optimizer = load_checkpoint(artifact_dir, policy, optimizer)
        policy.to(device)
    elif cfg.resume_ckpt is not None:
        logger.info(colored(f"Resuming from local checkpoint {cfg.resume_ckpt}", "cyan"))
        start_step, policy, optimizer = load_checkpoint(Path(cfg.resume_ckpt), policy, optimizer)
        policy.to(device)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    # Use the run cache directory for outputs instead of the default
    output_dir = run_cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    eval_env = None  # type: ignore  # will hold the evaluation environment if created

    # Track the best evaluation success-rate achieved so far (used for early-stopping style checkpointing)
    best_success_rate = 0.0

    if cfg.rollout_freq and cfg.eval_env:
        # ------------------------------------------------------------------
        # Create the DexMimicGen evaluation environment
        # ------------------------------------------------------------------
        device_str = "cpu" if cfg.device == "cpu" else "cuda"

        # Check if this is a DexMimicGen environment
        dexmimicgen_envs = [
            "TwoArmCoffee",
            "TwoArmThreading",
            "TwoArmThreePieceAssembly",
            "TwoArmTransport",
            "TwoArmLiftTray",
            "TwoArmBoxCleanup",
            "TwoArmDrawerCleanup",
            "TwoArmPouring",
            "TwoArmCanSortRandom",
        ]
        robomimic_envs = [
            "Lift",
            "Can",
            "Square",
            "Transport",
        ]
        mimicgen_envs = [
            "Threading",  # Single-arm threading task from MimicGen
        ]

        envs = dexmimicgen_envs + robomimic_envs + mimicgen_envs

        if cfg.eval_env in envs:
            # Create evaluation environment (vectorized or single based on debug flag)
            logger.info(f"Creating evaluation environment: {cfg.eval_env}")

            if cfg.debug:
                # Debug mode: use synchronous vectorized environment for easier debugging
                logger.info("Debug mode enabled: using synchronous vectorized environment")
                eval_env = create_vectorized_env(
                    env_name=cfg.eval_env,
                    num_envs=cfg.eval_num_envs,
                    device=device_str,
                    camera_size=cfg.eval_camera_size,
                    render_size=cfg.eval_render_size,
                    video_key=cfg.eval_video_key,
                    debug=True,
                )
            else:
                # Production mode: use asynchronous multiprocessing environment for speed
                logger.info("Production mode: using asynchronous multiprocessing environment")
                eval_env = create_vectorized_env(
                    env_name=cfg.eval_env,
                    num_envs=cfg.eval_num_envs,
                    device=device_str,
                    camera_size=cfg.eval_camera_size,
                    render_size=cfg.eval_render_size,
                    video_key=cfg.eval_video_key,
                    debug=False,
                )
        else:
            raise ValueError(f"Unknown environment: {cfg.eval_env}. Supported environments are: {envs}")

    while step < cfg.steps:
        # ------------------------------------------------------------------
        # Measure data loading time ----------------------------------------
        # ------------------------------------------------------------------
        iter_start_t = time.perf_counter()

        data_t0 = time.perf_counter()
        batch: dict[str, Any] = next(dl_iter)

        # Move tensors to device
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)
        data_load_ms = (time.perf_counter() - data_t0) * 1000

        # ------------------------------------------------------------------
        # Measure policy update time ---------------------------------------
        # ------------------------------------------------------------------
        update_t0 = time.perf_counter()

        loss, _ = policy.forward(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        update_ms = (time.perf_counter() - update_t0) * 1000

        iter_ms = (time.perf_counter() - iter_start_t) * 1000

        # ------------------------------------------------------------------
        # Logging ----------------------------------------------------------
        # ------------------------------------------------------------------
        loss_val = loss.item()

        if step % cfg.log_freq == 0:
            msg = (
                f"[step {step:>6d}/{cfg.steps}]"
                f" loss: {loss_val:.4f}"
                f" | data: {data_load_ms:.1f} ms"
                f" | update: {update_ms:.1f} ms"
                f" | iter: {iter_ms:.1f} ms"
            )
            logger.info(msg)
            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "time/data_load_ms": data_load_ms,
                        "time/update_ms": update_ms,
                        "time/iter_ms": iter_ms,
                    },
                    step=step,
                )

        # Checkpointing ----------------------------------------------
        if (step % cfg.save_freq == 0 and step != start_step) or step + 1 == cfg.steps:
            # 1) Save model-only weights (space-efficient, keeps history)
            #    └── policy_step_<n>
            model_dir = output_dir / f"policy_step_{step}"
            model_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(model_dir / "policy")

            # 2) Save full training state once under a constant "latest" directory
            latest_dir = output_dir / "latest"
            if latest_dir.exists():
                # Remove previous to avoid stale files
                shutil.rmtree(latest_dir)
            save_checkpoint(latest_dir, step, policy, optimizer)

            logger.info(
                colored(
                    f"Checkpoint saved (model-only @ {model_dir}, full state @ {latest_dir})",
                    "magenta",
                )
            )

            if wandb is not None:
                # Log model-only artifact (no optimizer)
                art_model = wandb.Artifact(name=f"run_{wandb.run.id}_model_step_{step}", type="model")
                art_model.add_dir(str(model_dir))
                wandb.log_artifact(art_model)

                # Log/overwrite the "latest" artifact with full state for resume
                art_latest = wandb.Artifact(name=f"run_{wandb.run.id}_latest", type="model")
                art_latest.add_dir(str(latest_dir))
                wandb.log_artifact(art_latest, aliases=["latest"])

        step += 1

        # ------------------------------------------------------------------
        # Rollout evaluation -----------------------------------------------
        # ------------------------------------------------------------------
        if (
            cfg.rollout_freq is not None
            and cfg.eval_env is not None
            and step % cfg.rollout_freq == 0
            and step != start_step
        ):
            rollout_t0 = time.perf_counter()

            success_rate, video_path, final_fps = _run_rollouts(
                policy=policy,
                env=eval_env,
                save_dir=output_dir,
                step=step,
                num_episodes=cfg.eval_num_episodes,
                run_start_time=run_start_time,
            )

            rollout_ms = (time.perf_counter() - rollout_t0) * 1000

            logger.info(
                colored(
                    f"[step {step:>6d}] eval success-rate: {success_rate * 100:.1f}% | "
                    f"rollout: {rollout_ms / 1000:.2f} s | {final_fps:.1f} fps",
                    "cyan",
                )
            )

            if wandb is not None:
                wandb.log(
                    {
                        "eval/success_rate": success_rate,
                        "time/rollout_ms": rollout_ms,
                    },
                    step=step,
                )
                if video_path is not None and video_path.exists():
                    fps = eval_env.fps
                    wandb.log({"eval/rollout_video": wandb.Video(str(video_path), format="mp4", fps=fps)}, step=step)

            # -------------------------------------------------------------
            # Checkpoint the model whenever we obtain a new best success-rate
            # -------------------------------------------------------------
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                logger.info(colored(f"New best success-rate! Saving checkpoint at step {step}", "magenta"))

                # 1) Save model-only weights for lightweight history
                best_model_dir = output_dir / f"best_step_{step}"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(best_model_dir / "policy")

                # 2) Save the full training state (includes model weights) under a constant
                #    "best" directory, overwriting the previous best each time.
                best_dir = output_dir / "best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                save_checkpoint(best_dir, step, policy, optimizer)

                if wandb is not None:
                    # Overwrite/refresh the "best" artifact so that the most recent best checkpoint is easy to retrieve
                    art_best = wandb.Artifact(name=f"run_{wandb.run.id}_best", type="model")
                    art_best.add_dir(str(best_dir))
                    wandb.log_artifact(art_best, aliases=["best", "latest"])

    logger.info(colored("Training finished!", "green", attrs=["bold"]))
    if wandb is not None:
        wandb.finish()

    if eval_env is not None:
        eval_env.close()

    # ---------------------------------------------------------------------
    # Cleanup --------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Clean up entire run directory after successful completion (videos/logs are saved to wandb)
    if run_cache_dir.exists():
        logger.info(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)
        logger.info("Run directory cleaned up successfully.")


if __name__ == "__main__":
    """
    Example commands:

    DexMimicGen/MimicGen/Robomimic environment training with high-resolution video recording and parallel environments:

    # Production mode (default): uses asynchronous multiprocessing for faster evaluation
    python -m resfit.lerobot.scripts.train_bc_dexmg \
        --dataset ankile/dexmimicgen-coffee-v1 \
        --policy act \
        --batch_size 64 --num_workers 8 \
        --wandb_project dexmimicgen-test \
        --rollout_freq 1000 --eval_env TwoArmCoffee \
        --eval_num_envs 8 \
        --eval_num_episodes 20 \
        --eval_camera_size 84 --eval_render_size 224

    # Debug mode: uses synchronous environments for easier debugging with ipdb
    python -m resfit.lerobot.scripts.train_bc_dexmg \
        --dataset ankile/dexmimicgen-coffee-v1 \
        --policy act \
        --batch_size 64 --num_workers 8 \
        --wandb_project dexmimicgen-test \
        --rollout_freq 1000 --eval_env TwoArmCoffee \
        --eval_num_envs 4 \
        --eval_num_episodes 12 \
        --eval_camera_size 84 --eval_render_size 224 \
        --debug

    # Train without proprioceptive observations (vision-only)
    python -m resfit.lerobot.scripts.train_bc_dexmg \
        --dataset ankile/dexmimicgen-coffee-v1 \
        --policy act \
        --batch_size 64 --num_workers 8 \
        --wandb_project dexmimicgen-test \
        --disable_proprioceptive_obs

    IMPORTANT: Parallel Environment Implementation
    =============================================

    This script supports two modes for running evaluation rollouts:

    1. **Production Mode (default)**: Uses AsyncVecEnv with multiprocessing
       - Environments run in separate processes for maximum parallelization
       - Faster evaluation but harder to debug
       - Recommended for training runs

    2. **Debug Mode (--debug flag)**: Uses SyncVecEnv
       - Environments run sequentially in the main process
       - Easier to debug with ipdb and breakpoints
       - Recommended for development and debugging

    The --eval_num_envs argument controls how many parallel environments to use:
    - Higher values = faster evaluation but more memory usage
    - Recommended: 4-8 for most setups
    - Use lower values (2-4) in debug mode to reduce complexity

    The --eval_num_episodes argument controls the total number of episodes to evaluate:
    - More episodes = better statistics but longer evaluation time
    - Recommended: 10-20 for development, 50+ for final evaluation
    - Episodes will be distributed across the parallel environments

    IMPORTANT: Observation Space Alignment
    =====================================
    1. Image dimensions: --eval_camera_size should match your dataset
       (e.g., 84 for 84x84 images, 224 for 224x224 images)

    2. High-resolution video recording: --eval_render_size adds an additional
       high-resolution camera using Robosuite's native multi-camera support.
       The policy still evaluates on standard resolution, but videos are captured
       at higher resolution for better visualization:
       - --eval_camera_size 84 --eval_render_size 224 for policy on 84x84, videos at 224x224
       - --eval_camera_size 84 --eval_render_size 448 for policy on 84x84, videos at 448x448
       If --eval_render_size equals --eval_camera_size, no additional camera is created.

       The system automatically selects an appropriate high-viewpoint camera (like
       'frontview' or 'birdview') for video recording while keeping dataset cameras
       at standard resolution for policy evaluation.

    3. Image keys: The rollout environment automatically uses the same
       cameras as your dataset conversion:
       - Panda envs: agentview, robot0_eye_in_hand, robot1_eye_in_hand
       - Transport: adds shouldercamera0, shouldercamera1
       - Humanoid: agentview, robot0_eye_in_left_hand, robot0_eye_in_right_hand
       - Can sort: frontview, robot0_eye_in_left_hand, robot0_eye_in_right_hand

       When --eval_render_size > --eval_camera_size, an additional high-resolution
       camera (e.g., frontview or birdview) is automatically selected and used for video recording.

    4. State observations: Uses end-effector poses, quaternions, and gripper
       positions (same as dataset conversion), NOT joint positions.
    """
    main(args_cli)
