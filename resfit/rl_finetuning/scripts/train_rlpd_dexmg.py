# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os

# Cap all BLAS/OpenMP threadpools (critical to set before importing numpy/torch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Stop threads from spin-waiting
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")


import hashlib
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, RandomSampler, ReplayBuffer
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.config.rlpd import RLPDDexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.checkpoint import save_checkpoint
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.hugging_face import _hf_download_buffer, _hf_upload_buffer
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform

# -----------------------------------------------------------------------------
# Hugging Face buffer cache helpers (global) ---------------------------------
# -----------------------------------------------------------------------------
OFFLINE_HF_REPO = os.environ.get("HF_OFFLINE_BUFFER_REPO", None)
ONLINE_HF_REPO = os.environ.get("HF_ONLINE_BUFFER_REPO", None)

# Generic environment variable (shared across algorithms) -------------------
# ``CACHE_DIR`` specifies the root folder for **all** local caches.
# Falls back to the current directory if unset.
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()

# Dedicated sub-folders for the different cache types -----------------------
RLPD_OFFLINE_CACHE_DIR = _CACHE_ROOT / "offline_buffer_cache"
RLPD_ONLINE_CACHE_DIR = _CACHE_ROOT / "online_buffer_cache"


# -----------------------------------------------------------------------------
# Repository-local imports ------------------------------------------------------
# -----------------------------------------------------------------------------

os.environ["MUJOCO_GL"] = "egl"

if "MUJOCO_EGL_DEVICE_ID" in os.environ:
    del os.environ["MUJOCO_EGL_DEVICE_ID"]


def _add_transitions_to_buffer(
    *,
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict,
    device: torch.device,
    image_keys: list[str],
    lowdim_keys: list[str],
    num_envs: int,
    online_rb: ReplayBuffer,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and converts images to uint8 for storage.
    """
    for i in range(num_envs):
        # Handle terminal observation (same logic as main loop)
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
        elif done[i]:
            continue  # no transition for this one
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # Keep only relevant keys & convert images to uint8 for storage
        obs_keys_set = set(image_keys) | set(lowdim_keys)
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
            },
            batch_size=[],
        ).unsqueeze(0)

        online_rb.add(td)


# -----------------------------------------------------------------------------
# Main training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------
def main(cfg: RLPDDexmgConfig):
    # Validate config: MultiStep transform with multiple environments is broken
    if cfg.algo.n_step > 1 and cfg.num_envs > 1:
        raise ValueError(
            f"MultiStep transform (n_step={cfg.algo.n_step}) with multiple environments (num_envs={cfg.num_envs}) "
            "is not supported due to data interleaving issues. Please use either:\n"
            "  - n_step=1 (single-step) with any num_envs, or\n"
            "  - num_envs=1 with any n_step"
        )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------------
    # Environment setup ----------------------------------------------------
    # ---------------------------------------------------------------------
    raw_env = create_vectorized_env(
        env_name=cfg.task,
        num_envs=cfg.num_envs,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
    )
    cfg.eval_num_envs = min(cfg.eval_num_envs, cfg.eval_num_episodes)
    num_cpus_available = os.cpu_count() - 1 if os.cpu_count() is not None else 1
    cfg.eval_num_envs = min(num_cpus_available, cfg.eval_num_envs)

    # Seeding ----------------------------------------------------------------
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # ---------------------------------------------------------------------
    # wandb / logging -----------------------------------------------------
    # ---------------------------------------------------------------------

    _hp_parts: list[str] = [
        cfg.task,  # e.g. "TwoArmBoxCleanup"
        f"n{cfg.algo.n_step}",  # n-step horizon
        f"utd{cfg.algo.num_updates_per_iteration}",  # updates-to-data ratio
        f"buf{cfg.algo.buffer_size}",  # replay buffer size
    ]

    # Offline dataset statistics (if any)
    if cfg.offline_data is not None and cfg.offline_data.num_episodes is not None:
        _hp_parts.append(f"off{cfg.offline_data.num_episodes}ep")

    # Learning-rate, expressed in scientific notation for brevity (e.g. 1e-4 → 1e-04)
    _hp_parts.append(f"lr{cfg.agent.actor_lr:.0e}")

    # Additional flags ---------------------------------------------------------
    if cfg.agent.clip_q_target_to_reward_range:
        _hp_parts.append("clipT")

    hp_str = "_".join(_hp_parts)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"
    if cfg.wandb.name is not None:
        run_name = cfg.wandb.name + "_" + run_name

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )

    # Create a timestamped folder in CACHE_DIR for all outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"run_{timestamp}_{run_name}"

    # Create subdirectories for models and outputs
    model_save_dir = run_cache_dir / "models"
    outputs_dir = run_cache_dir / "outputs"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Observation / action dimensions -------------------------------------
    # ---------------------------------------------------------------------
    # Determine which image keys (camera observations) will be used. The
    # configuration can specify either a single camera name (str) or a list of
    # names.
    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    assert isinstance(image_keys, list)
    lowdim_dim = raw_env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = raw_env.observation_space[image_keys[0]].shape[1:]
    action_dim = raw_env.action_space.shape[1]

    lowdim_keys = ["observation.state"]

    # ---------------------------------------------------------------------
    # Networks ------------------------------------------------------------
    # ---------------------------------------------------------------------
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
    )
    horizon = raw_env.vec_env.metadata["horizon"]

    # ---------------------------------------------------------------------
    # Action and State Normalization Setup --------------------------------
    # ---------------------------------------------------------------------
    # Load dataset to get normalization statistics
    print("Loading dataset for normalization statistics...")
    dataset = LeRobotDataset(cfg.offline_data.name)

    # Create action scaler from dataset statistics
    # Note: For standard RLPD, we use action_scale=0.0 for the ActionScaler (no range expansion)
    # but the actor itself uses action_scale=1.0 (from config) to produce full-scale actions
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"],
        action_scale=0.0,  # No expansion for standard RL (unlike residual RL)
        min_range_per_dim=1e-1,  # Same safeguard as residual RL
        device=device,
    )

    # Create state standardizer from dataset statistics
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=1e-1,  # Same safeguard as residual RL
        device=device,
    )

    # Helper function to normalize observations from environment
    def normalize_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply state normalization to environment observations."""
        normalized_obs = obs.copy()
        normalized_obs["observation.state"] = state_standardizer.standardize(obs["observation.state"])
        return normalized_obs

    # Helper function to denormalize actions for environment
    def denormalize_action(normalized_action: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions [-1, 1] back to environment action space."""
        return action_scaler.unscale(normalized_action)

    # Create a wrapper for the evaluation environment that applies normalization
    class NormalizedEnvWrapper:
        def __init__(self, env, normalize_obs_fn, denormalize_action_fn):
            self.env = env
            self.normalize_obs = normalize_obs_fn
            self.denormalize_action = denormalize_action_fn
            # Forward all attributes to the wrapped environment
            for attr in ["num_envs", "observation_space", "action_space", "render", "fps"]:
                if hasattr(env, attr):
                    setattr(self, attr, getattr(env, attr))

        def reset(self):
            raw_obs, info = self.env.reset()
            normalized_obs = self.normalize_obs(raw_obs)
            return normalized_obs, info

        def step(self, normalized_actions):
            # Denormalize actions for environment
            env_actions = self.denormalize_action(normalized_actions)
            raw_next_obs, reward, terminated, truncated, info = self.env.step(env_actions)
            normalized_next_obs = self.normalize_obs(raw_next_obs)
            return normalized_next_obs, reward, terminated, truncated, info

    # Create training environment with normalization wrapper
    env = NormalizedEnvWrapper(
        env=raw_env,
        normalize_obs_fn=normalize_obs,
        denormalize_action_fn=denormalize_action,
    )

    # Create evaluation environment with normalization wrapper
    raw_eval_env = create_vectorized_env(
        env_name=cfg.task,
        num_envs=cfg.eval_num_envs,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
    )

    eval_env = NormalizedEnvWrapper(
        env=raw_eval_env,
        normalize_obs_fn=normalize_obs,
        denormalize_action_fn=denormalize_action,
    )

    # ---------------------------------------------------------------------
    # Replay buffers -------------------------------------------------------
    # ---------------------------------------------------------------------
    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    online_rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu", ndim=1),
        sampler=RandomSampler(),
        # Add a MultiStep transform so that the buffer stores n-step returns
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=online_batch_size,
    )

    # Simple online buffer caching for warmup
    online_cache_meta = {
        "task": cfg.task,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "horizon": horizon,
        "warmup_size": cfg.algo.learning_starts,
        "capacity": cfg.algo.buffer_size,
        # Include normalization parameters in cache key
        "action_scale": 0.0,
        "min_action_range": 1e-1,
        "min_state_std": 1e-1,
        "normalization_version": "v1",
    }
    _online_meta_str = json.dumps(online_cache_meta, sort_keys=True)
    online_cache_hash = hashlib.sha1(_online_meta_str.encode()).hexdigest()[:8]  # noqa: S324
    online_cache_dir = RLPD_ONLINE_CACHE_DIR / online_cache_hash

    # Try to download from HF
    if ONLINE_HF_REPO is not None:
        dl_dir = _hf_download_buffer(ONLINE_HF_REPO, online_cache_hash, RLPD_ONLINE_CACHE_DIR)
        if dl_dir is not None:
            online_cache_dir = dl_dir

    loaded_online_from_cache = False
    if online_cache_dir.exists():
        online_rb.loads(online_cache_dir)
        loaded_online_from_cache = True
        print(f"Loaded warmup buffer from cache (size={len(online_rb)})")

    assert cfg.offline_data is not None and cfg.offline_data.num_episodes is not None

    offline_rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=horizon * cfg.offline_data.num_episodes, device="cpu", ndim=1),
        sampler=RandomSampler(),
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        batch_size=offline_batch_size,
        prefetch=cfg.algo.prefetch_batches,
    )

    # ------------------------------------------------------------------
    # Convert offline dataset episodes into transitions and fill buffer
    # ------------------------------------------------------------------

    def _populate_offline_buffer(
        dataset_name: str,
        rb: ReplayBuffer,
        image_keys: list[str],
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
        num_episodes: int | None = None,
    ) -> int:
        """Iterates through *dataset* sequentially, converts consecutive frames
        into (s, a, r, s', done) transitions and pushes them into *rb*.

        Returns the number of transitions added.
        """
        dataset = LeRobotDataset(dataset_name)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        episode_cache: dict[int, dict] = {}
        transitions = 0

        for sample in tqdm(loader, desc="Processing offline dataset"):
            # note: this goes through each episode in order
            # within each episode, it goes through steps in order
            # and `frame_idx` indicates the step number within each episode
            ep_idx = int(sample["episode_index"].item())
            if num_episodes is not None and ep_idx == num_episodes:
                break

            # ------------------------------------------------------------------
            # Build observation for the *current* frame ------------------------
            # ------------------------------------------------------------------
            obs_state = sample["observation.state"].float().squeeze(0)  # (obs_dim,)
            # Apply state standardization
            obs_state = state_standardizer.standardize(obs_state)

            curr_obs = {
                "observation.state": obs_state,
            }
            for k in image_keys:
                curr_obs[k] = sample[k].squeeze(0)  # (c, h, w) float 0 to 1

            # Convert images to uint8 for memory-efficient storage
            to_uint8(curr_obs, image_keys)

            # Apply action scaling to normalize to [-1, 1]
            curr_action = sample["action"].float().squeeze(0)  # (ac_dim,)
            curr_action = action_scaler.scale(curr_action)

            # "next.done" tells us if the **next** frame will terminate the episode
            done_flag = bool(sample["next.done"].item())

            # ------------------------------------------------------------------
            # If we already cached the *previous* frame for this episode we can
            # create a transition now.
            # ------------------------------------------------------------------
            if ep_idx in episode_cache:
                prev = episode_cache[ep_idx]

                transition = TensorDict(
                    {
                        "obs": TensorDict(prev["obs"], batch_size=[]),
                        "action": prev["action"],
                        "next": TensorDict(
                            {
                                "obs": TensorDict(curr_obs, batch_size=[]),
                                "done": torch.tensor(done_flag, dtype=torch.bool),
                                "reward": torch.tensor(float(done_flag), dtype=torch.float32),
                            },
                            batch_size=[],
                        ),
                    },
                    batch_size=[],
                ).unsqueeze(0)

                rb.add(transition)

                transitions += 1

            # Cache current frame for pairing with the next one ---------------
            episode_cache[ep_idx] = {
                "obs": curr_obs,
                "action": curr_action,
                "done": done_flag,
                "step_id": sample["frame_index"].item(),
            }

        return transitions

    # ------------------------------------------------------------------
    # Caching layer for offline replay buffer ---------------------------
    # ------------------------------------------------------------------
    # Build a metadata dictionary that uniquely identifies the buffer
    cache_meta = {
        "task": cfg.task,
        "dataset_name": cfg.offline_data.name,
        "num_episodes": cfg.offline_data.num_episodes,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "horizon": horizon,
        # Include normalization parameters in cache key
        "action_scale": 0.0,
        "min_action_range": 1e-1,
        "min_state_std": 1e-1,
        "normalization_version": "v1",  # Increment this if normalization logic changes
    }

    # Deterministically hash the metadata to create a short cache directory name
    meta_str = json.dumps(cache_meta, sort_keys=True)
    cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324

    # Base local path for this buffer ---------------------------------------
    cache_dir = RLPD_OFFLINE_CACHE_DIR / cache_hash

    # Try to download/extract from the Hub (will no-op if file not there)
    downloaded_dir = None
    if OFFLINE_HF_REPO is not None:
        downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, RLPD_OFFLINE_CACHE_DIR)
    if downloaded_dir is not None:
        cache_dir = downloaded_dir  # use extracted location

    loaded_from_cache = False
    if cache_dir.exists():
        try:
            offline_rb.loads(cache_dir)
            loaded_from_cache = True
            print(f"Loaded offline buffer from cache at {cache_dir} (size={len(offline_rb)})")
        except Exception as e:
            print(f"Failed to load offline buffer from cache ({e}). Regenerating…")

    if not loaded_from_cache:
        added = _populate_offline_buffer(
            dataset_name=cfg.offline_data.name,
            rb=offline_rb,
            image_keys=image_keys,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            num_episodes=cfg.offline_data.num_episodes,
        )
        print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")

        # Save buffer to disk for future runs + upload to Hub ----------------
        cache_dir.mkdir(parents=True, exist_ok=True)
        offline_rb.dumps(cache_dir)

        with open(cache_dir / "user_metadata.json", "w") as f:
            json.dump(cache_meta, f, indent=2)

        if OFFLINE_HF_REPO is not None:
            _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)
    else:
        added = len(offline_rb)

    # ------------------------------------------------------------------
    # Warm-up phase (random policy) --------------------------------------
    # ------------------------------------------------------------------
    if len(online_rb) < cfg.algo.learning_starts and not loaded_online_from_cache:
        print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")
        obs, _ = env.reset()
        # --------------------------------------------------------------
        # Logging helper: print progress every 1 000 collected transitions
        # --------------------------------------------------------------
        next_log_threshold = 1000  # first threshold for progress message

        while len(online_rb) < cfg.algo.learning_starts:
            rand_actions = torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1  # uniform in [-1,1]
            next_obs, reward, terminated, truncated, info = env.step(rand_actions)
            done = terminated | truncated

            _add_transitions_to_buffer(
                obs=obs,
                next_obs=next_obs,
                actions=rand_actions,  # Store normalized actions in buffer
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
            )

            # ----------------------------------------------------------
            # Progress logging (every ~1 000 transitions) --------------
            # ----------------------------------------------------------
            if len(online_rb) >= next_log_threshold:
                print(f"[Warm-up] {len(online_rb)} / {cfg.algo.learning_starts} transitions collected")
                next_log_threshold += 1000

            obs = next_obs  # roll state

        # Persist freshly-collected buffer (local + HF) --------------------
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        online_rb.dumps(online_cache_dir)
        with open(online_cache_dir / "user_metadata.json", "w") as f:
            json.dump(online_cache_meta, f, indent=2)
        if ONLINE_HF_REPO is not None:
            _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up done. Online buffer size = {len(online_rb)} transitions")

        loaded_online_from_cache = True  # treat as cached going forward

    obs, _ = env.reset()

    # global_step = len(online_rb)  # already includes warm-up or cached size
    global_step = 0  # Let's not count the warm-up phase for training
    best_eval_success_rate = 0.0
    eval_metrics = {}  # Initialize eval metrics
    training_cum_time = 0.0
    train_start_time = time.time()

    while global_step < cfg.algo.total_timesteps:
        iter_start = time.time()
        # ------------------------------------------------------------------
        # (1) Collect action ------------------------------------------------
        # ------------------------------------------------------------------
        with torch.no_grad(), utils.eval_mode(agent):
            stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
            action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

        # ------------------------------------------------------------------
        # (2) Environment step ---------------------------------------------
        # ------------------------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # Add to online replay buffer --------------------------------------
        _add_transitions_to_buffer(
            obs=obs,
            next_obs=next_obs,
            actions=action,  # Store normalized actions in buffer
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
        )
        obs = next_obs  # roll

        # ------------------------------------------------------------------
        # (3) Periodic evaluation ------------------------------------------
        # ------------------------------------------------------------------
        if (
            (global_step > 0 and global_step % cfg.eval_interval_every_steps == 0)
            or (cfg.eval_first and global_step == 0)
            or (global_step + cfg.num_envs >= cfg.algo.total_timesteps)
        ):
            eval_metrics = run_dexmg_evaluation(
                env=eval_env,
                agent=agent,
                num_episodes=cfg.eval_num_episodes,
                device=device,
                global_step=global_step,
                save_video=cfg.save_video,
                save_q_plots=cfg.save_video,  # Enable Q-plots when video saving is enabled
                run_name=run_name,
                output_dir=outputs_dir,
            )

            # Handle model saving when success rate improves
            current_success_rate = eval_metrics["eval/success_rate"]
            if current_success_rate > best_eval_success_rate:
                print(f"🎉 New best success rate: {current_success_rate:.4f} (prev: {best_eval_success_rate:.4f})")
                best_eval_success_rate = current_success_rate

                # Save checkpoint using utility function
                ckpt_path = model_save_dir / "best.pt"
                save_checkpoint(
                    agent=agent,
                    checkpoint_path=ckpt_path,
                    global_step=global_step,
                    config=cfg,
                    success_rate=current_success_rate,
                )

                if wandb.run is not None:
                    wandb.save(str(ckpt_path))

        global_step += cfg.num_envs
        # ------------------------------------------------------------------
        # (4) Updates -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.algo.update_every_n_steps == 0:
            for i in range(cfg.algo.num_updates_per_iteration):
                # --------------------------------------------------------------
                # Sample mixed online/offline batch
                # --------------------------------------------------------------
                online_batch = (
                    online_rb.sample(int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction)))
                    .copy()
                    .to(device, non_blocking=True)
                )
                offline_batch = (
                    offline_rb.sample(int(cfg.algo.batch_size * cfg.algo.offline_fraction))
                    .copy()
                    .to(device, non_blocking=True)
                )

                batch = torch.cat([online_batch, offline_batch], dim=0)

                update_actor = i == cfg.algo.num_updates_per_iteration - 1

                bc_batch = None
                if update_actor and cfg.agent.bc_loss_coef > 0:
                    bc_batch = offline_batch

                metrics = agent.update(batch, stddev, update_actor, bc_batch, ref_agent=agent)

                # Add batch data metrics
                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

            # Step LR schedulers during warmup
            if global_step < cfg.agent.lr_warmup_steps:
                agent.step_lr_schedulers()

        training_cum_time += time.time() - iter_start

        # ------------------------------------------------------------------
        # (6) Logging -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.log_freq == 0:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

            # Prepare base logging dict
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                "timing/training_total_time": time.time() - train_start_time,
                "timing/aggregate_steps_per_second": global_step / (time.time() - train_start_time),
                # Log current learning rates
                "lr/actor": agent.actor_opt.param_groups[0]["lr"],
                "lr/critic": agent.critic_opt.param_groups[0]["lr"],
            }

            # Add metrics, filtering out internal data
            filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
            log_dict.update(filtered_metrics)

            # Compute action statistics only when logging (for regular RLPD, these are the policy actions)
            if "_actions" in metrics:
                actions = metrics["_actions"]
                # Compute L1/L2 magnitudes (only during logging to save computation)
                action_l1_magnitude = torch.mean(torch.abs(actions)).item()
                action_l2_magnitude = torch.mean(torch.square(actions)).item()

                log_dict["train/action_l1_magnitude"] = action_l1_magnitude
                log_dict["train/action_l2_magnitude"] = action_l2_magnitude
                log_dict["histograms/actions"] = wandb.Histogram(actions.numpy().reshape(-1))
            else:
                action_l1_magnitude = None
                action_l2_magnitude = None

            # Add Q values histogram when available
            if "_target_q" in metrics:
                target_q = metrics["_target_q"]
                log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))

            # Only log encoder LR if encoder is not frozen
            if not getattr(cfg.agent, "freeze_encoder", False):
                log_dict["lr/encoder"] = agent.encoder_opt.param_groups[0]["lr"]

            wandb.log(log_dict, step=global_step)

            # Enhanced print statement with action magnitudes, gradient norms, and actor LR
            current_actor_lr = agent.actor_opt.param_groups[0]["lr"]

            actor_loss_str = f"actor_loss_base={metrics['train/actor_loss_base']:.4f}"

            print_msg = (
                f"[{global_step}] {actor_loss_str} "
                f"critic_loss={metrics['train/critic_loss']:.4f} "
                f"actor_lr={current_actor_lr:.2e}"
            )
            if action_l1_magnitude is not None and action_l2_magnitude is not None:
                print_msg += f" action_l1={action_l1_magnitude:.4f} action_l2={action_l2_magnitude:.4f}"

            # Add gradient norms to print statement
            if "train/actor_grad_norm" in metrics:
                print_msg += f" actor_grad_norm={metrics['train/actor_grad_norm']:.4f}"

            # Add L2 penalty if active
            if "train/actor_l2_penalty" in metrics:
                print_msg += f" l2_penalty={metrics['train/actor_l2_penalty']:.4f}"

            print(print_msg)

        # LR schedulers --------------------------------------------------------

    print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")

    # Clean up entire run directory after successful completion (videos/logs are saved to wandb)
    if run_cache_dir.exists():
        print(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)
        print("Run directory cleaned up successfully.")


# -----------------------------------------------------------------------------
# Hydra entry point -----------------------------------------------------------
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_name="rlpd_dexmg_config")
def hydra_entry(cfg: RLPDDexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
