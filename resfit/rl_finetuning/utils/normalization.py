# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Action scaling utilities for residual RL training.

This module provides a clean interface for min-max scaling of actions to [-1, 1] range,
with proper safeguards against numerical instabilities.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class ActionScaler:
    """
    Handles min-max scaling of actions to [-1, 1] range for residual RL training.

    This class encapsulates all the logic for:
    - Computing action limits from dataset statistics
    - Applying safeguards against numerical instabilities
    - Scaling actions to [-1, 1] and unscaling back to original range
    - Handling device placement automatically
    """

    class Limits(NamedTuple):
        """Action limits for scaling."""

        min: torch.Tensor
        max: torch.Tensor

    def __init__(
        self,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        action_scale: float = 1.0,
        min_range_per_dim: float = 1e-1,
        device: torch.device | str = "cpu",
    ):
        """
        Initialize the action scaler.

        Args:
            action_min: Minimum values from dataset statistics
            action_max: Maximum values from dataset statistics
            action_scale: Scale factor to expand the action range (1 + action_scale)
            min_range_per_dim: Minimum range per dimension to prevent normalization blow-up
            device: Device to place tensors on
        """
        self.device = torch.device(device)
        self.action_scale = action_scale
        self.min_range_per_dim = min_range_per_dim

        # Move inputs to device
        action_min = action_min.to(self.device)
        action_max = action_max.to(self.device)

        # Compute action center and half-range
        action_mid = (action_min + action_max) / 2
        action_half_range = (action_max - action_min) / 2

        # Apply safeguard: ensure minimum range to prevent normalization blow-up
        min_half_range = torch.tensor(min_range_per_dim / 2, device=self.device)
        action_half_range = torch.maximum(action_half_range, min_half_range)

        # Expand the range by action_scale factor
        expanded_half_range = action_half_range * (1 + action_scale)

        # Store the final limits
        self._limits = self.Limits(
            min=action_mid - expanded_half_range,
            max=action_mid + expanded_half_range,
        )

        # Precompute range for efficiency (with safeguard)
        self._range = self._limits.max - self._limits.min
        self._range = torch.maximum(self._range, torch.tensor(1e-8, device=self.device))

        print("ActionScaler initialized:")
        print(f"  Original range: [{action_min.min():.4f}, {action_max.max():.4f}]")
        print(f"  Expanded range: [{self._limits.min.min():.4f}, {self._limits.max.max():.4f}]")
        print(f"  Action scale factor: {action_scale}")

    @property
    def limits(self) -> Limits:
        """Get the action limits."""
        return self._limits

    def scale(self, action: torch.Tensor) -> torch.Tensor:
        """
        Scale action to [-1, 1] range.

        Args:
            action: Action tensor to scale

        Returns:
            Scaled action in [-1, 1] range
        """
        # Move limits to same device as input action
        action_min = self._limits.min.to(action.device)
        action_max = self._limits.max.to(action.device)
        range_vals = self._range.to(action.device)

        # Clamp input to prevent extreme values
        action_clamped = torch.clamp(action, action_min, action_max)

        # Scale to [-1, 1]
        return 2.0 * (action_clamped - action_min) / range_vals - 1.0

    def unscale(self, scaled_action: torch.Tensor) -> torch.Tensor:
        """
        Unscale action from [-1, 1] back to original range.

        Args:
            scaled_action: Scaled action in [-1, 1] range

        Returns:
            Unscaled action in original range
        """
        # Move limits to same device as input action
        action_min = self._limits.min.to(scaled_action.device)
        action_max = self._limits.max.to(scaled_action.device)

        # Clamp to [-1, 1] to prevent extreme unscaled values
        scaled_clamped = torch.clamp(scaled_action, -1.0, 1.0)

        # Unscale back to original range
        return action_min + (scaled_clamped + 1.0) * (action_max - action_min) / 2.0

    def to(self, device: torch.device | str) -> ActionScaler:
        """Move scaler to a different device."""
        new_scaler = ActionScaler.__new__(ActionScaler)
        new_scaler.device = torch.device(device)
        new_scaler.action_scale = self.action_scale
        new_scaler.min_range_per_dim = self.min_range_per_dim
        new_scaler._limits = self.Limits(
            min=self._limits.min.to(device),
            max=self._limits.max.to(device),
        )
        new_scaler._range = self._range.to(device)
        return new_scaler

    @classmethod
    def from_dataset_stats(
        cls,
        action_stats: dict,
        action_scale: float = 1.0,
        min_range_per_dim: float = 1e-1,
        device: torch.device | str = "cpu",
    ) -> ActionScaler:
        """
        Create ActionScaler from dataset statistics.

        Args:
            action_stats: Dictionary with 'min' and 'max' keys containing action statistics
            action_scale: Scale factor to expand the action range
            min_range_per_dim: Minimum range per dimension to prevent normalization blow-up
            device: Device to place tensors on

        Returns:
            Configured ActionScaler instance
        """
        action_min = torch.tensor(action_stats["min"], dtype=torch.float32)
        action_max = torch.tensor(action_stats["max"], dtype=torch.float32)

        return cls(
            action_min=action_min,
            action_max=action_max,
            action_scale=action_scale,
            min_range_per_dim=min_range_per_dim,
            device=device,
        )


class StateStandardizer:
    """
    Handles standardization of states to mean=0, std=1.

    This class provides a clean interface for state normalization with safeguards.
    """

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        min_std: float = 1e-1,
        device: torch.device | str = "cpu",
    ):
        """
        Initialize the state standardizer.

        Args:
            state_mean: Mean values from dataset statistics
            state_std: Standard deviation values from dataset statistics
            min_std: Minimum std to prevent normalization blow-up
            device: Device to place tensors on
        """
        self.device = torch.device(device)
        self.min_std = min_std

        # Move to device and apply safeguards
        self._mean = state_mean.to(self.device)
        self._std = torch.maximum(state_std.to(self.device), torch.tensor(min_std, device=self.device))

        print("StateStandardizer initialized:")
        print(f"  Mean range: [{self._mean.min():.4f}, {self._mean.max():.4f}]")
        print(f"  Std range: [{self._std.min():.6f}, {self._std.max():.6f}]")

    def standardize(self, state: torch.Tensor) -> torch.Tensor:
        """
        Standardize state to mean=0, std=1.

        Args:
            state: State tensor to standardize

        Returns:
            Standardized state
        """
        # Move stats to same device as input state
        mean = self._mean.to(state.device)
        std = self._std.to(state.device)

        # Additional runtime safeguard
        std_safe = torch.maximum(std, torch.tensor(1e-8, device=state.device))

        return (state - mean) / std_safe

    def to(self, device: torch.device | str) -> StateStandardizer:
        """Move standardizer to a different device."""
        new_standardizer = StateStandardizer.__new__(StateStandardizer)
        new_standardizer.device = torch.device(device)
        new_standardizer.min_std = self.min_std
        new_standardizer._mean = self._mean.to(device)
        new_standardizer._std = self._std.to(device)
        return new_standardizer

    @classmethod
    def from_dataset_stats(
        cls,
        state_stats: dict,
        min_std: float = 1e-1,
        device: torch.device | str = "cpu",
    ) -> StateStandardizer:
        """
        Create StateStandardizer from dataset statistics.

        Args:
            state_stats: Dictionary with 'mean' and 'std' keys containing state statistics
            min_std: Minimum std to prevent normalization blow-up
            device: Device to place tensors on

        Returns:
            Configured StateStandardizer instance
        """
        state_mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
        state_std = torch.tensor(state_stats["std"], dtype=torch.float32)

        return cls(
            state_mean=state_mean,
            state_std=state_std,
            min_std=min_std,
            device=device,
        )
