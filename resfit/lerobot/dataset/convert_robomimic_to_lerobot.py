# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

#!/usr/bin/env python3
"""
Script to convert Robomimic HDF5 trajectory data into LeRobot dataset format.

Usage:
  python convert_robomimic_to_lerobot.py --dataset /path/to/robomimic_dataset.hdf5 \\
    --output_dir /path/to/lerobot_dataset --repo_id [your-hf-account]/dataset-name
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_info
from tqdm import tqdm


def get_env_metadata_from_dataset(dataset_path: str) -> dict[str, Any]:
    """
    Retrieves env metadata from robomimic dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:
            'env_name': name of environment
            'type': type of environment
            'env_kwargs': dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    return env_meta  # noqa: RET504


def get_dataset_trajectories(
    dataset_path: str,
    filter_key: str | None = None,
    max_episodes: int | None = None,
    exclude_episodes: list[int] | None = None,
) -> list[str]:
    """
    Get list of trajectory keys from the dataset.

    Args:
        dataset_path (str): path to dataset
        filter_key (str, optional): filter key to select subset of trajectories
        max_episodes (int, optional): maximum number of episodes to include
        exclude_episodes (list[int], optional): list of 1-indexed episode numbers to exclude

    Returns:
        List[str]: sorted list of trajectory keys (e.g., ['demo_0', 'demo_1', ...])
    """
    with h5py.File(dataset_path, "r") as f:
        if filter_key is not None:
            demos = [
                elem.decode("utf-8") if isinstance(elem, bytes) else elem for elem in np.array(f[f"mask/{filter_key}"])
            ]
        else:
            demos = list(f["data"].keys())

    # Sort in increasing number order
    inds = np.argsort([int(elem[5:]) for elem in demos])  # demo_N -> N
    demos = [demos[i] for i in inds]

    # Filter out excluded episodes (convert from 1-indexed to 0-indexed)
    if exclude_episodes is not None:
        # Convert 1-indexed episode numbers to 0-indexed for filtering
        demos = [demo for i, demo in enumerate(demos, start=1) if i not in exclude_episodes]

    if max_episodes is not None and max_episodes > 0:
        demos = demos[:max_episodes]

    return demos


def analyze_dataset_structure(dataset_path: str, demo_keys: list[str]) -> tuple[dict[str, Any], list[str], int]:
    """
    Analyze the dataset structure to determine features and image keys.

    Args:
        dataset_path (str): path to dataset
        demo_keys (List[str]): list of demonstration keys

    Returns:
        Tuple containing:
        - features dict for LeRobot dataset
        - list of image observation keys
        - fps (frames per second)
    """
    with h5py.File(dataset_path, "r") as f:
        # Get first demo for structure analysis
        first_demo = demo_keys[0]
        demo_grp = f[f"data/{first_demo}"]

        # Get basic trajectory info (some keys might not exist)
        actions = demo_grp["actions"][()]

        # Analyze observation structure
        obs_grp = demo_grp["obs"]
        obs_keys = list(obs_grp.keys())

        # Get environment metadata to determine expected image keys
        env_meta = json.loads(f["data"].attrs["env_args"])
        env_name = env_meta.get("env_name", "")

        # Define expected image keys based on environment name
        expected_image_keys = get_expected_image_keys(env_name)

        # Define expected low_dim_keys based on environment name
        expected_low_dim_keys = get_expected_low_dim_keys(env_name)

        # Separate image and non-image observations
        image_keys = []
        state_keys = []

        for key in obs_keys:
            obs_data = obs_grp[key]
            if len(obs_data.shape) == 4:  # (T, H, W, C) - image data
                image_keys.append(key)
            else:  # non-image observations
                state_keys.append(key)

        # Filter image keys to only include expected ones
        if expected_image_keys:
            image_keys = [key for key in image_keys if key in expected_image_keys]

        # Filter state keys to only include expected low_dim_keys
        if expected_low_dim_keys:
            state_keys = [key for key in state_keys if key in expected_low_dim_keys]

        # Build features dict
        features = {}

        # Action feature
        features["action"] = {
            "dtype": "float32",
            "shape": actions.shape[1:],
            "names": get_action_names(env_name, actions.shape[1]),
        }

        # Done feature - create from episode length if dones doesn't exist
        features["next.done"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        }

        # State observations (concatenate only expected low_dim_keys)
        if state_keys:
            state_components = []
            state_names = []
            for key in state_keys:
                if key in obs_grp:  # Make sure the key exists in the dataset
                    obs_data = obs_grp[key][0]  # First timestep
                    if obs_data.ndim == 0:  # scalar
                        obs_data = np.array([obs_data])
                    state_components.append(obs_data)
                    if obs_data.shape == ():
                        state_names.append(key)
                    else:
                        state_names.extend([f"{key}_{i}" for i in range(len(obs_data))])
                else:
                    print(f"Warning: Expected low_dim_key '{key}' not found in dataset observations")

            if state_components:
                concatenated_state = np.concatenate(state_components)
                features["observation.state"] = {
                    "dtype": "float32",
                    "shape": concatenated_state.shape,
                    "names": state_names,
                }

        # Image features
        for img_key in image_keys:
            img_data = obs_grp[img_key]
            # Remove '_image' suffix if present for cleaner naming
            clean_key = img_key.replace("_image", "")
            features[f"observation.images.{clean_key}"] = {
                "dtype": "video",
                "shape": img_data.shape[1:],  # (H, W, C)
                "names": ["height", "width", "channel"],
            }

    # Default fps - robomimic datasets typically don't store fps info
    fps = 20

    return features, image_keys, fps


def get_expected_image_keys(env_name: str) -> list[str]:
    """
    Get expected image keys based on environment name from generate_training_config.py

    Args:
        env_name (str): Environment name

    Returns:
        List[str]: List of expected image observation keys
    """
    # Mapping based on generate_training_config.py
    panda_image_keys = [
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot1_eye_in_hand_image",
    ]

    panda_transport_image_keys = [
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot1_eye_in_hand_image",
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

    # Map environment names to image keys
    env_lower = env_name.lower()
    if "transport" in env_lower:
        return panda_transport_image_keys
    if "cansort" in env_lower or "can_sort" in env_lower:  # Check both variants
        return humanoid_can_sort_image_keys
    if any(task in env_lower for task in ["pouring", "coffee"]):
        return humanoid_image_keys

    # Default to panda keys for other tasks
    return panda_image_keys


def get_expected_low_dim_keys(env_name: str) -> list[str]:
    """
    Get expected low_dim_keys based on environment name from generate_training_config.py

    Args:
        env_name (str): Environment name

    Returns:
        List[str]: List of expected low-dimensional observation keys
    """
    # Mapping based on generate_training_config.py
    panda_low_dim_keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
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

    # Map environment names to low_dim_keys
    env_lower = env_name.lower()
    if any(task in env_lower for task in ["pouring", "coffee"]) or "cansort" in env_lower or "can_sort" in env_lower:
        return humanoid_low_dim_keys
    # Default to panda keys for other tasks
    return panda_low_dim_keys


def get_action_names(env_name: str, action_dim: int) -> list[str]:
    """
    Get meaningful action names based on environment name and action dimension.

    Based on the action annotations from the DexMimicGen wrapper:
    - 0-2: Right wrist Δpos Cartesian x/y/z offset
    - 3-5: Right wrist Δrot Axis-angle components rx,ry,rz
    - 6-11: Right Inspire-hand joints
    - 12-14: Left wrist Δpos Cartesian x/y/z offset
    - 15-17: Left wrist Δrot Axis-angle components for left EE
    - 18-23: Left Inspire-hand joints

    Args:
        env_name (str): Environment name
        action_dim (int): Total action dimension

    Returns:
        List[str]: List of meaningful action names
    """
    env_lower = env_name.lower()

    # Check if this is a dexterous hand environment (PandaDex or similar)
    is_dexterous = any(task in env_lower for task in ["lifttray", "boxcleanup", "drawercleanup"])

    # Check if this is a humanoid environment
    is_humanoid = any(task in env_lower for task in ["pouring", "coffee", "cansort", "can_sort"])

    # Check if this is a single-arm environment
    single_arm_tasks = ["lift", "can", "pickplacecan", "square", "nutassemblysquare", "threading"]
    is_single_arm = any(task in env_lower for task in single_arm_tasks)

    if is_dexterous and action_dim == 24:
        # 24-DOF dexterous bimanual actions
        return [
            "right_wrist_delta_pos_x",
            "right_wrist_delta_pos_y",
            "right_wrist_delta_pos_z",  # 0-2: Right wrist position
            "right_wrist_delta_rot_rx",
            "right_wrist_delta_rot_ry",
            "right_wrist_delta_rot_rz",  # 3-5: Right wrist rotation
            "right_thumb_flexion",
            "right_thumb_opposition",
            "right_index_flexion",  # 6-8: Right thumb and index
            "right_middle_flexion",
            "right_ring_flexion",
            "right_pinky_flexion",  # 9-11: Right middle, ring, pinky
            "left_wrist_delta_pos_x",
            "left_wrist_delta_pos_y",
            "left_wrist_delta_pos_z",  # 12-14: Left wrist position
            "left_wrist_delta_rot_rx",
            "left_wrist_delta_rot_ry",
            "left_wrist_delta_rot_rz",  # 15-17: Left wrist rotation
            "left_thumb_flexion",
            "left_thumb_opposition",
            "left_index_flexion",  # 18-20: Left thumb and index
            "left_middle_flexion",
            "left_ring_flexion",
            "left_pinky_flexion",  # 21-23: Left middle, ring, pinky
        ]
    if is_humanoid and action_dim == 24:
        # 24-DOF humanoid bimanual actions (similar structure but for humanoid)
        return [
            "right_eef_delta_pos_x",
            "right_eef_delta_pos_y",
            "right_eef_delta_pos_z",  # 0-2: Right end-effector position
            "right_eef_delta_rot_rx",
            "right_eef_delta_rot_ry",
            "right_eef_delta_rot_rz",  # 3-5: Right end-effector rotation
            "right_thumb_flexion",
            "right_thumb_opposition",
            "right_index_flexion",  # 6-8: Right thumb and index
            "right_middle_flexion",
            "right_ring_flexion",
            "right_pinky_flexion",  # 9-11: Right middle, ring, pinky
            "left_eef_delta_pos_x",
            "left_eef_delta_pos_y",
            "left_eef_delta_pos_z",  # 12-14: Left end-effector position
            "left_eef_delta_rot_rx",
            "left_eef_delta_rot_ry",
            "left_eef_delta_rot_rz",  # 15-17: Left end-effector rotation
            "left_thumb_flexion",
            "left_thumb_opposition",
            "left_index_flexion",  # 18-20: Left thumb and index
            "left_middle_flexion",
            "left_ring_flexion",
            "left_pinky_flexion",  # 21-23: Left middle, ring, pinky
        ]
    if action_dim == 14:
        # Standard bimanual Panda actions (7 DOF per arm)
        return [
            "robot0_eef_delta_pos_x",
            "robot0_eef_delta_pos_y",
            "robot0_eef_delta_pos_z",  # 0-2: Robot0 end-effector position
            "robot0_eef_delta_rot_rx",
            "robot0_eef_delta_rot_ry",
            "robot0_eef_delta_rot_rz",  # 3-5: Robot0 end-effector rotation
            "robot0_gripper_action",  # 6: Robot0 gripper
            "robot1_eef_delta_pos_x",
            "robot1_eef_delta_pos_y",
            "robot1_eef_delta_pos_z",  # 7-9: Robot1 end-effector position
            "robot1_eef_delta_rot_rx",
            "robot1_eef_delta_rot_ry",
            "robot1_eef_delta_rot_rz",  # 10-12: Robot1 end-effector rotation
            "robot1_gripper_action",  # 13: Robot1 gripper
        ]
    if is_single_arm and action_dim == 7:
        # Single-arm Panda actions (7 DOF) - for Lift, Can, Square, Threading tasks
        return [
            "eef_delta_pos_x",
            "eef_delta_pos_y",
            "eef_delta_pos_z",  # 0-2: End-effector position
            "eef_delta_rot_rx",
            "eef_delta_rot_ry",
            "eef_delta_rot_rz",  # 3-5: End-effector rotation (axis-angle)
            "gripper_action",  # 6: Gripper open/close
        ]
    if action_dim == 20:
        # Potential 20-DOF configuration (could be dual-arm with different setup)
        # This might be for environments with different gripper configurations
        return [
            "robot0_eef_delta_pos_x",
            "robot0_eef_delta_pos_y",
            "robot0_eef_delta_pos_z",  # 0-2: Robot0 position
            "robot0_eef_delta_rot_rx",
            "robot0_eef_delta_rot_ry",
            "robot0_eef_delta_rot_rz",  # 3-5: Robot0 rotation
            "robot0_finger_0",
            "robot0_finger_1",
            "robot0_finger_2",
            "robot0_finger_3",  # 6-9: Robot0 multi-finger gripper
            "robot1_eef_delta_pos_x",
            "robot1_eef_delta_pos_y",
            "robot1_eef_delta_pos_z",  # 10-12: Robot1 position
            "robot1_eef_delta_rot_rx",
            "robot1_eef_delta_rot_ry",
            "robot1_eef_delta_rot_rz",  # 13-15: Robot1 rotation
            "robot1_finger_0",
            "robot1_finger_1",
            "robot1_finger_2",
            "robot1_finger_3",  # 16-19: Robot1 multi-finger gripper
        ]
    if action_dim == 12:
        # 12-DOF configuration (could be dual-arm with 6 DOF each)
        return [
            "robot0_eef_delta_pos_x",
            "robot0_eef_delta_pos_y",
            "robot0_eef_delta_pos_z",  # 0-2: Robot0 position
            "robot0_eef_delta_rot_rx",
            "robot0_eef_delta_rot_ry",
            "robot0_eef_delta_rot_rz",  # 3-5: Robot0 rotation
            "robot1_eef_delta_pos_x",
            "robot1_eef_delta_pos_y",
            "robot1_eef_delta_pos_z",  # 6-8: Robot1 position
            "robot1_eef_delta_rot_rx",
            "robot1_eef_delta_rot_ry",
            "robot1_eef_delta_rot_rz",  # 9-11: Robot1 rotation
        ]
    if action_dim == 6:
        # Single-arm 6-DOF (position + rotation, no gripper)
        return [
            "eef_delta_pos_x",
            "eef_delta_pos_y",
            "eef_delta_pos_z",  # 0-2: End-effector position
            "eef_delta_rot_rx",
            "eef_delta_rot_ry",
            "eef_delta_rot_rz",  # 3-5: End-effector rotation
        ]
    if action_dim == 4:
        # Simplified 4-DOF (position + gripper)
        return [
            "eef_delta_pos_x",
            "eef_delta_pos_y",
            "eef_delta_pos_z",  # 0-2: End-effector position
            "gripper_action",  # 3: Gripper
        ]
    # Fail if action dimension is not supported
    raise ValueError(f"Action dimension {action_dim} not supported")


def convert_robomimic_to_lerobot(
    dataset_path: str,
    output_dir: str,
    repo_id: str | None = None,
    filter_key: str | None = None,
    train_ratio: float = 1.0,
    max_episodes: int | None = None,
    exclude_episodes: list[int] | None = None,
):
    """
    Convert robomimic HDF5 dataset to LeRobot format.

    Args:
        dataset_path (str): Path to robomimic HDF5 dataset
        output_dir (str): Output directory for LeRobot dataset
        repo_id (str, optional): HuggingFace repository ID for uploading
        filter_key (str, optional): Filter key to select subset of trajectories
        train_ratio (float): Ratio of data to use for training (rest goes to test)
        max_episodes (int, optional): Maximum number of episodes to convert
        exclude_episodes (list[int], optional): List of 1-indexed episode numbers to exclude from conversion
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Get environment metadata
    env_meta = get_env_metadata_from_dataset(dataset_path)
    env_name = env_meta.get("env_name", "RobomimicEnv")

    # Get trajectory list
    demo_keys = get_dataset_trajectories(dataset_path, filter_key, max_episodes, exclude_episodes)
    print(f"Found {len(demo_keys)} trajectories to convert")

    # Analyze dataset structure
    features, image_keys, fps = analyze_dataset_structure(dataset_path, demo_keys)

    # Create LeRobot dataset
    dataset_repo_id = repo_id if repo_id else Path(output_dir).name
    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=fps,
        root=str(output_dir),
        robot_type="robomimic",
        features=features,
        use_videos=len(image_keys) > 0,
    )

    # Process trajectories
    with h5py.File(dataset_path, "r") as f:
        for demo_key in tqdm(demo_keys, desc="Processing trajectories"):
            demo_grp = f[f"data/{demo_key}"]

            # Load trajectory data
            actions = demo_grp["actions"][()]

            # Handle optional data
            dones = demo_grp["dones"][()] if "dones" in demo_grp else None

            obs_grp = demo_grp["obs"]

            num_frames = len(actions)

            # Create dones array if it doesn't exist (only last frame is done)
            if dones is None:
                dones = np.zeros(num_frames, dtype=bool)
                dones[-1] = True  # Last frame is done

            # Process each frame
            for frame_idx in range(num_frames):
                task = env_name
                frame_data = {
                    "action": np.array(actions[frame_idx], dtype=np.float32).reshape(features["action"]["shape"]),
                    "next.done": np.array([dones[frame_idx]], dtype=bool).reshape(features["next.done"]["shape"]),
                }

                # Add state observations
                if "observation.state" in features:
                    # Get expected low_dim_keys for this environment
                    expected_low_dim_keys = get_expected_low_dim_keys(env_name)

                    state_components = []
                    for key in expected_low_dim_keys:
                        if key in obs_grp and len(obs_grp[key].shape) != 4:  # Not an image and exists
                            obs_data = obs_grp[key][frame_idx]
                            if obs_data.ndim == 0:  # scalar
                                obs_data = np.array([obs_data])
                            state_components.append(obs_data)

                    if state_components:
                        concatenated_state = np.concatenate(state_components)
                        frame_data["observation.state"] = np.array(concatenated_state, dtype=np.float32).reshape(
                            features["observation.state"]["shape"]
                        )

                # Add image observations
                for img_key in image_keys:
                    clean_key = img_key.replace("_image", "")
                    feature_key = f"observation.images.{clean_key}"
                    if feature_key in features:
                        img_data = obs_grp[img_key][frame_idx]
                        # Ensure uint8 format for images
                        if img_data.dtype != np.uint8:
                            img_data = (
                                (img_data * 255).astype(np.uint8)
                                if img_data.max() <= 1.0
                                else img_data.astype(np.uint8)
                            )
                        frame_data[feature_key] = img_data

                dataset.add_frame(frame=frame_data, task=task)

            dataset.save_episode()

    # Create train/test split if needed
    total_episodes = dataset.meta.info["total_episodes"]
    if train_ratio < 1.0:
        num_train = int(total_episodes * train_ratio)
        train_range = f"0:{num_train}"
        test_range = f"{num_train}:{total_episodes}"
        dataset.meta.info["splits"] = {"train": train_range, "test": test_range}
        write_info(dataset.meta.info, dataset.root)

    print(f"Successfully created LeRobot dataset at {output_dir}")

    # Upload to HF Hub if requested
    if repo_id:
        dataset.push_to_hub()
        print(f"Dataset successfully uploaded to {repo_id}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert robomimic HDF5 trajectories to LeRobot format.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to robomimic HDF5 dataset file",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LeRobot dataset")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID for uploading the dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="Filter key to select subset of trajectories from the dataset",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Ratio of trajectories to assign to train split (0.0-1.0). Remaining trajectories assigned to test split.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert. Converts all if not specified.",
    )
    parser.add_argument(
        "--exclude-episodes",
        type=int,
        nargs="*",
        default=None,
        help="List of 1-indexed episode numbers to exclude from conversion (e.g., --exclude-episodes 1 3 5)",
    )

    args = parser.parse_args()

    convert_robomimic_to_lerobot(
        args.dataset,
        args.output_dir,
        args.repo_id,
        args.filter_key,
        args.train_ratio,
        args.max_episodes,
        args.exclude_episodes,
    )
