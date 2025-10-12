# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import logging
import os

import gymnasium as gym

# Import mimicgen to register MimicGen environments with robosuite
import numpy as np

# Suppress robosuite's verbose INFO / WARNING logs at import time
logging.getLogger("robosuite").setLevel(logging.ERROR)

import robosuite  # noqa: E402
import torch  # noqa: E402
from robosuite import load_composite_controller_config, macros  # noqa: E402
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv  # noqa: E402

macros.IMAGE_CONVENTION = "opencv"


# Mapping from (canonical) environment name to the corresponding list of robot models
# NOTE: When adding new tasks, always reference the *actual* robosuite environment
# class name (the one expected by `robosuite.make`).
ENV_ROBOTS = {
    # ------------------------------------------------------------------
    # DexMimicGen multi-arm tasks
    # ------------------------------------------------------------------
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1FixedLowerBody"],
    # ------------------------------------------------------------------
    # MimicGen single-arm tasks
    # ------------------------------------------------------------------
    # Threading task -- single Panda arm from MimicGen
    "Threading": ["Panda"],
    # ------------------------------------------------------------------
    # Robomimic benchmark tasks (single-arm unless otherwise noted)
    # ------------------------------------------------------------------
    # Lift task -- single Panda arm
    "Lift": ["Panda"],
    # Can task -- implemented in robosuite as `PickPlaceCan`
    "PickPlaceCan": ["Panda"],
    # Square task -- implemented in robosuite as `NutAssemblySquare`
    "NutAssemblySquare": ["Panda"],
}
# Create a named logger
logger = logging.getLogger(__name__)


class RobosuiteGymWrapper:
    """
    Gym-like wrapper for robosuite environments to make them compatible with the training script.

    Robosuite environments use the old gym API (step returns 4 values) and have different
    observation/action interfaces, so this wrapper adapts them to work with our training loop.

    Idxs	Meaning (all values are per-time-step targets, sent at 20 Hz)
    0-2	Right wrist Δpos Cartesian x / y / z offset (metres) for the EE site gripper0_right_grip_site.
    3-5	Right wrist Δrot Axis-angle components rx,ry,rzrx,ry,rz; ‖r‖ = rotation angle (rad).
    6-11	Right Inspire-hand joints (joint-position targets, rad)
    6 Thumb flexion
    7 Thumb roll / opposition
    8 Index flexion
    9 Middle flexion
    10 Ring flexion
    11 Pinky flexion
    12-14	Left wrist Δpos Cartesian x / y / z offset for gripper0_left_grip_site.
    15-17	Left wrist Δrot Axis-angle components for left EE orientation.
    18-23	Left Inspire-hand joints (same ordering as right).


    """

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        render_gpu_device_id: int = 0,
        camera_size: int = 84,
        render_size: int | None = None,
        env_id: int = 0,
    ):
        # ------------------------------------------------------------------
        # Allow common aliases used in the Robomimic literature.
        # These are mapped to the actual robosuite environment names.
        # ------------------------------------------------------------------
        alias_map = {
            # Robomimic papers / datasets refer to these tasks without the
            # full robosuite class name. We translate them here so that
            # callers can simply pass "Lift", "Can", "Square", or
            # "Transport" and things will work out of the box.
            "Can": "PickPlaceCan",
            "Square": "NutAssemblySquare",
            "Transport": "TwoArmTransport",
        }

        # Preserve original user-provided name for logging / heuristics
        self.original_env_name = env_name
        # Resolve to the canonical robosuite env name (if alias exists)
        env_name = alias_map.get(env_name, env_name)

        self.env_name = env_name
        self.num_envs = num_envs
        self.render_gpu_device_id = render_gpu_device_id
        self.camera_size = camera_size
        self.render_size = render_size if render_size is not None else (240, 320)
        self.env_id = env_id

        # ------------------------------------------------------------------
        # Episode horizon -- override for some long-running DexMimicGen tasks.
        # For all other tasks we fall back to robosuite's default of 1000.
        # ------------------------------------------------------------------
        self.horizon = {
            "Lift": 100,
            "PickPlaceCan": 200,
            "NutAssemblySquare": 300,
            "Threading": 500,
            "TwoArmTransport": 800,
            "TwoArmBoxCleanup": 300,
            "TwoArmCoffee": 400,
            "TwoArmLiftTray": 650,
            "TwoArmPouring": 400,
            "TwoArmThreePieceAssembly": 500,
            "TwoArmThreading": 300,
            "TwoArmCanSortRandom": 400,
        }.get(env_name, 1000)

        # Add Gymnasium-required attributes
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 20, "horizon": self.horizon}
        self.spec = None  # Not required for vectorization
        self.render_mode = "rgb_array"  # Default render mode for camera observations

        # Store video camera key for rendering (will be set by create_vectorized_env)
        self.video_key = None

        self.episode_steps = 0

        if env_name not in ENV_ROBOTS:
            raise ValueError(f"Unknown robosuite environment: {env_name}")

        # Only import DexMimimcGen if needed
        if env_name in [
            "TwoArmPouring",
            "TwoArmLiftTray",
            "TwoArmDrawerCleanup",
            "TwoArmBoxCleanup",
            "TwoArmCoffee",
            "TwoArmThreePieceAssembly",
            "TwoArmThreading",
            "TwoArmCanSortRandom",
        ]:
            import dexmimicgen  # noqa: F401, PLC0415

        robots = ENV_ROBOTS[env_name]

        # Get expected image keys for this environment (same as dataset conversion)
        expected_image_keys = self._get_expected_image_keys(env_name)
        # Remove '_image' suffix to get camera names for robosuite
        camera_names = [key.replace("_image", "") for key in expected_image_keys]

        self.expected_image_keys = expected_image_keys  # Store for use in _process_obs

        # Create environment using robosuite.make()
        env_kwargs = {
            "env_name": env_name,
            "robots": robots,
            "controller_configs": load_composite_controller_config(robot=robots[0]),
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": False,
            "use_camera_obs": True,
            "control_freq": 20,
            "camera_names": camera_names,
            "camera_heights": self.camera_size,
            "camera_widths": self.camera_size,
            "horizon": self.horizon,
            "renderer": "mujoco",
            "render_gpu_device_id": self.render_gpu_device_id,
        }

        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.render_gpu_device_id)

        # Only log once per vectorized group (env_id==0) and at DEBUG level
        if self.env_id == 0:
            logger.debug(f"render_gpu_device_id: {self.render_gpu_device_id}")

        # NOTE: This is a crucial change for the rollouts to work -- should it live here or elsewhere?
        if "composite_controller_specific_configs" in env_kwargs["controller_configs"]:
            env_kwargs["controller_configs"]["composite_controller_specific_configs"]["ik_input_ref_frame"] = "world"

        if env_name == "TwoArmTransport":
            env_kwargs["env_configuration"] = "opposed"
            env_kwargs["controller_configs"]["body_parts"]["right"]["input_ref_frame"] = "world"

        self.env: ManipulationEnv = robosuite.make(**env_kwargs)

        logger.debug(
            f"Successfully created {env_name} environment via robosuite.make() "
            f"with cameras at {camera_size}x{camera_size}"
        )
        logger.debug(f"Configured cameras: {camera_names}")

        # The `num_envs` argument is supported for legacy reasons, will remove at some point soon
        if num_envs != 1:
            raise ValueError("This wrapper is for 1 env, multiple envs are supported by the vectorized env wrapper")

        # Define action and observation spaces after environment creation
        self._setup_spaces()

    def _setup_spaces(self):
        """Setup observation and action spaces for Gymnasium compatibility."""
        # Get action space dimensions from the robosuite environment
        action_dim = self.env.action_dim
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Get a sample observation to determine actual shapes
        sample_obs_raw = self.env.reset()
        sample_obs = self._process_obs_for_space_inference(sample_obs_raw)

        # Setup observation space based on actual observation shapes
        obs_spaces = {}
        for key, value in sample_obs.items():
            obs_spaces[key] = gym.spaces.Box(
                low=-np.inf if "state" in key else 0.0,
                high=np.inf if "state" in key else 1.0,
                shape=value.shape,
                dtype=value.dtype,
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _process_obs_for_space_inference(self, obs):
        """Process observations for space inference without storing _last_obs."""
        processed_obs = {}

        # Extract robot state
        expected_low_dim_keys = self._get_expected_low_dim_keys(self.env_name)
        state_components = []

        for key in expected_low_dim_keys:
            if key in obs:
                obs_data = obs[key]
                if obs_data.ndim == 0:  # scalar
                    obs_data = np.array([obs_data])
                state_components.append(obs_data)

        if state_components:
            concatenated_state = np.concatenate(state_components)
            processed_obs["observation.state"] = concatenated_state.astype(np.float32)

        # Extract camera observations
        for img_key in self.expected_image_keys:
            camera_name = img_key.replace("_image", "")
            robosuite_key = f"{camera_name}_image"

            if robosuite_key in obs:
                img = obs[robosuite_key]
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

                clean_key = img_key.replace("_image", "")
                processed_obs[f"observation.images.{clean_key}"] = img

        return processed_obs

    def seed(self, seed=None):
        """Seed the environment's random number generator."""
        # For robosuite environments, we don't have direct seeding control
        # This is a no-op for compatibility
        return [seed]

    def reset(self, *, seed=None, options=None):
        """Reset the environment and return initial observation."""
        # Gymnasium interface: reset can accept seed and options
        # For robosuite environments, we'll ignore these for now
        obs = self.env.reset()
        processed_obs = self._process_obs(obs)
        self._last_obs = processed_obs  # Store for video recording
        self.episode_steps = 0
        return processed_obs, {}

    def step(self, action):
        """Step the environment with the given action."""
        # Convert action from torch tensor to numpy if needed
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action[0]  # Take first action if batched

        obs, reward, done, info = self.env.step(action)
        self.episode_steps += 1
        # Convert to the expected format
        processed_obs = self._process_obs(obs)

        # Return scalar values - Gymnasium will handle device placement and batching in vectorized env
        reward_scalar = float(reward)

        # Terminate episode on success (reward == 1) to shortcircuit successful rollouts
        success = reward == 1.0
        terminated_scalar = bool(success)
        truncated_scalar = bool(done)  # Robosuite returns done when timeout

        if terminated_scalar or truncated_scalar:
            info = {
                **info,
                "success": success,
                "episode_steps": self.episode_steps,
            }
            self.episode_steps = 0

        return processed_obs, reward_scalar, terminated_scalar, truncated_scalar, info

    def _process_obs(self, obs):
        """Process robosuite observations to match expected format."""
        processed_obs = {}

        # Extract robot state using the same features as dataset conversion
        state_components = []

        # Get expected low_dim_keys for this environment (same as dataset conversion)
        expected_low_dim_keys = self._get_expected_low_dim_keys(self.env_name)

        for key in expected_low_dim_keys:
            if key in obs:
                obs_data = obs[key]
                if obs_data.ndim == 0:  # scalar
                    obs_data = np.array([obs_data])
                state_components.append(obs_data)
            else:
                logger.debug(f"Expected state key '{key}' not found in environment observations")

        if state_components:
            concatenated_state = np.concatenate(state_components)
            # Return numpy array - Gymnasium will handle device placement and batching
            processed_obs["observation.state"] = concatenated_state.astype(np.float32)

        # Extract camera observations using the same logic as dataset conversion
        for img_key in self.expected_image_keys:
            # Convert to robosuite camera name (remove '_image' suffix)
            camera_name = img_key.replace("_image", "")
            robosuite_key = f"{camera_name}_image"

            # Robosuite images are (H, W, C) in uint8, need (C, H, W) in float32
            if robosuite_key in obs:
                img = obs[robosuite_key]
                img = img.astype(np.float32) / 255.0  # Convert to float32 and normalize
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

                # Use the same naming convention as dataset conversion: observation.images.{clean_key}
                clean_key = img_key.replace("_image", "")
                processed_obs[f"observation.images.{clean_key}"] = img

        # Store last obs for video recording (convert back to torch tensors on device)
        self._last_obs = {}
        for key, value in processed_obs.items():
            self._last_obs[key] = value

        # Log observation keys for debugging (only on first call)
        if not hasattr(self, "_logged_obs_keys"):
            logger.debug(f"Created observation keys: {list(processed_obs.keys())}")
            self._logged_obs_keys = True

        return processed_obs

    def _get_expected_image_keys(self, env_name: str):
        """Return the expected image keys for a given environment.

        The logic mirrors what is used during dataset conversion so that the
        training / rollout code sees the exact same observation structure.
        """

        # ---------------------------------------------
        # Canonical camera key sets
        # ---------------------------------------------
        panda_image_keys_single = [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]

        panda_image_keys_multi = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot1_eye_in_hand_image",
        ]

        panda_transport_image_keys = [
            # "agentview_image",
            # "robot0_eye_in_hand_image",
            # "robot1_eye_in_hand_image",
            "shouldercamera0_image",
            "shouldercamera1_image",
        ]

        humanoid_image_keys = [
            "agentview_image",
            "robot0_eye_in_left_hand_image",
            "robot0_eye_in_right_hand_image",
        ]

        humanoid_can_sort_image_keys = [
            "frontview_image",
            "robot0_eye_in_left_hand_image",
            "robot0_eye_in_right_hand_image",
        ]

        env_lower = env_name.lower()

        # Transport (two-arm Panda)
        if "transport" in env_lower:
            return panda_transport_image_keys

        # Humanoid variants -------------------------------------------------
        if "cansort" in env_lower:
            return humanoid_can_sort_image_keys
        if any(task in env_lower for task in ["pouring", "coffee"]):
            return humanoid_image_keys

        # Single-arm Panda tasks (Lift, Can, Square, Threading, etc.) --------
        if env_lower in {"lift", "can", "pickplacecan", "square", "nutassemblysquare", "threading"}:
            return panda_image_keys_single

        # Fallback to two-arm Panda cameras --------------------------------
        return panda_image_keys_multi

    def _get_expected_low_dim_keys(self, env_name: str):
        """Return the expected low-dimensional state keys for a given environment."""

        panda_low_dim_keys_single = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]

        panda_low_dim_keys_multi = [
            *panda_low_dim_keys_single,
            "robot1_eef_pos",
            "robot1_eef_quat",
            "robot1_gripper_qpos",
        ]

        humanoid_low_dim_keys = [
            "robot0_right_eef_pos",
            "robot0_right_eef_quat",
            "robot0_right_gripper_qpos",
            "robot0_left_eef_pos",
            "robot0_left_eef_quat",
            "robot0_left_gripper_qpos",
        ]

        env_lower = env_name.lower()

        # Humanoid variants
        if any(task in env_lower for task in ["pouring", "coffee", "cansort"]):
            return humanoid_low_dim_keys

        # Single-arm Panda tasks
        if env_lower in {"lift", "can", "pickplacecan", "square", "nutassemblysquare", "threading"}:
            return panda_low_dim_keys_single

        # Default: two-arm Panda
        return panda_low_dim_keys_multi

    def render(self):
        """Return an RGB frame (H, W, 3, uint8) for video recording."""

        # Prefer the configured video_key's camera when available
        camera_name = "agentview"
        if getattr(self, "video_key", None):
            # Expect keys like 'observation.images.agentview' → extract last component
            key = self.video_key
            if isinstance(key, str) and "." in key:
                camera_name = key.split(".")[-1]
            elif isinstance(key, str):
                camera_name = key

        frame = self.env.sim.render(
            camera_name=camera_name,
            height=self.render_size[0],
            width=self.render_size[1],
        )[::-1]

        return frame  # noqa: RET504

    def set_video_key(self, video_key: str):
        """Set which observation key to use for video recording."""
        self.video_key = video_key
        # Reduce verbosity of underlying vectorized env loggers
        logging.getLogger("gymnasium.vector").setLevel(logging.ERROR)

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def unwrapped(self):
        """Access to the underlying robosuite environment."""
        return self

    def get_wrapper_attr(self, name: str):
        """Utility for Gymnasium AsyncVectorEnv compatibility.

        Gymnasium's AsyncVectorEnv workers rely on every environment (or wrapper) exposing
        a `get_wrapper_attr` method that walks through potential wrapper chains to
        retrieve attributes. Since this class is not derived from `gym.Wrapper`, we
        provide a minimal implementation that simply returns the attribute from this
        instance (if it exists) or raises an `AttributeError` otherwise. This is
        sufficient because the environment is not wrapped multiple times on the
        worker side.
        """
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def set_wrapper_attr(self, name: str, value):
        """Counterpart to `get_wrapper_attr` expected by Gymnasium.

        Allows the vectorised worker to set attributes on the environment even when
        it is not a `gym.Wrapper`.
        """
        setattr(self, name, value)


def make_dexmimicgen_env(
    env_name: str, camera_size: int = 84, render_size: int | None = None, render_gpu_device_id: int = 0, env_id: int = 0
):
    """Factory function to create a DexMimicGen environment for vectorization."""

    def _make():
        return RobosuiteGymWrapper(
            env_name=env_name,
            num_envs=1,
            render_gpu_device_id=render_gpu_device_id,
            camera_size=camera_size,
            render_size=render_size,
            env_id=env_id,
        )

    return _make


class VectorizedEnvWrapper:
    """Simple wrapper around gymnasium vectorized environments to add rendering capability."""

    def __init__(
        self, vec_env: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv, video_key: str, device: str = "cpu"
    ):
        self.vec_env = vec_env
        self.video_key = video_key
        self._last_obs = None
        self.device = device

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self._last_obs = obs
        obs = self._convert_obs_to_torch(obs, self.device)
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.vec_env.step(actions)
        self._last_obs = obs

        # Convert to torch tensors
        obs = self._convert_obs_to_torch(obs, self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool)

        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return RGB frames from all environments (num_envs, H, W, 3, uint8) for video recording."""
        frames: np.ndarray | None = self.vec_env.render()
        if frames is None:
            raise RuntimeError("No frames returned from vectorized environment")
        return frames

    @property
    def fps(self):
        return self.vec_env.metadata["render_fps"]

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)

    def _convert_obs_to_torch(self, obs, device):
        """Convert numpy observations from vectorized env to PyTorch tensors for policy."""
        if isinstance(obs, dict):
            torch_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    torch_obs[key] = torch.from_numpy(value).to(device)
                else:
                    torch_obs[key] = value
            return torch_obs
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).to(device)
        return obs


def create_vectorized_env(
    env_name: str,
    num_envs: int,
    device: str = "cpu",
    camera_size: int = 84,
    render_size: int | None = None,
    debug: bool = False,
    video_key: str = "observation.images.agentview",
) -> VectorizedEnvWrapper:
    """Create vectorized environment using Gymnasium's vector environments."""

    # Create list of environment factory functions
    env_fns = []

    # Get CUDA_VISIBLE_DEVICES as a list of device ordinals (as seen by torch)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if cuda_visible_devices is not None:
        # This is a comma-separated list of device ordinals as seen by the system
        visible_device_ids = [int(x) for x in cuda_visible_devices.split(",") if x.strip() != ""]
    else:
        # If not set, assume all devices from 0 to torch.cuda.device_count()-1 are visible
        visible_device_ids = list(range(torch.cuda.device_count()))

    num_visible_gpus = len(visible_device_ids) if visible_device_ids else 1

    for env_id in range(num_envs):
        if num_visible_gpus > 1:
            render_gpu_device_id = visible_device_ids[env_id % num_visible_gpus]
        else:
            render_gpu_device_id = visible_device_ids[0] if visible_device_ids else 0
        env_fns.append(make_dexmimicgen_env(env_name, camera_size, render_size, render_gpu_device_id, env_id))

    if debug:
        # Use synchronous vectorized environment for debugging
        logger.debug("Debug mode: using gymnasium.vector.SyncVectorEnv")
        vec_env = gym.vector.SyncVectorEnv(
            env_fns,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    else:
        # Use asynchronous vectorized environment for performance
        logger.debug("Production mode: using gymnasium.vector.AsyncVectorEnv")
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=True,  # Avoid shared memory issues with complex observations
            copy=True,  # Ensure observations are properly copied
            context="spawn",  # Use spawn for CUDA compatibility
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    # Propagate the desired video camera to all sub-environments
    # Each worker exposes `set_wrapper_attr(name, value)` which we call here
    vec_env.call("set_wrapper_attr", "video_key", video_key)

    # Wrap with our custom wrapper for rendering
    wrapped_env = VectorizedEnvWrapper(vec_env, video_key, device)

    # Add environment metadata for later retrieval
    wrapped_env.env_name = env_name
    wrapped_env.camera_size = camera_size
    wrapped_env.render_size = render_size

    logger.debug(f"Created {num_envs} vectorized {env_name} environments")
    logger.debug(f"Set video key to '{video_key}' for video recording")
    return wrapped_env
