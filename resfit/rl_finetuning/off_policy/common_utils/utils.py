# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import re

import numpy as np
import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal
from torchvision import transforms


def get_rescale_transform(target_size):
    return transforms.Resize(
        target_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )


def concat_obs(curr_idx, obses, obs_stack) -> torch.Tensor:
    """
    cat obs as [obses[curr_idx], obses[curr_idx-1], ... obs[curr_odx-obs_stack+1]]
    """
    vals = []
    for offset in range(obs_stack):
        if curr_idx - offset >= 0:
            val = obses[curr_idx - offset]
            if not isinstance(val, torch.Tensor):
                val = torch.from_numpy(val)
            vals.append(val)
        else:
            vals.append(torch.zeros_like(vals[-1]))
    return torch.cat(vals, dim=0)


class eval_mode:  # noqa: N801
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def orth_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def initialize_layer_weights(layer: nn.Linear, distribution: str, scale: float | None = None):
    """Initialize layer weights and bias using the specified distribution.

    Args:
        layer: The linear layer to initialize
        distribution: The distribution to use ('default', 'normal', 'orthogonal', 'xavier_uniform')
        scale: Scale parameter - for 'normal': std, for 'orthogonal'/'xavier_uniform': gain
    """
    if distribution == "default":
        # Use PyTorch's default initialization - do nothing
        pass
    elif distribution == "normal":
        if scale is not None:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=scale)
        else:
            # Use default normal initialization
            nn.init.normal_(layer.weight)
            if layer.bias is not None:
                nn.init.normal_(layer.bias)
    elif distribution == "orthogonal":
        gain = scale if scale is not None else 1.0
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif distribution == "xavier_uniform":
        gain = scale if scale is not None else 1.0
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        raise ValueError(
            f"Unsupported distribution: {distribution}. "
            f"Supported options: 'default', 'normal', 'orthogonal', 'xavier_uniform'"
        )


def apply_initialization_to_network(
    network: nn.Module, distribution: str, scale: float | None = None, exclude_final_layer: bool = False
):
    """Apply initialization to all Linear layers in a network.

    Args:
        network: The network to initialize
        distribution: The distribution to use
        scale: Scale parameter for the distribution
        exclude_final_layer: If True, skip the last Linear layer in the network
    """
    if distribution == "default":
        return  # Do nothing for default initialization

    linear_layers = [m for m in network.modules() if isinstance(m, nn.Linear)]
    layers_to_init = linear_layers[:-1] if exclude_final_layer and linear_layers else linear_layers

    for layer in layers_to_init:
        initialize_layer_weights(layer, distribution, scale)


def clip_action_norm(action, max_norm):
    assert max_norm > 0
    assert action.dim() == 2 and action.size(1) == 7

    ee_action = action[:, :6]
    gripper_action = action[:, 6:]

    ee_action_norm = ee_action.norm(dim=1).unsqueeze(1)
    ee_action = ee_action / ee_action_norm
    assert (ee_action.norm(dim=1).min() - 1).abs() <= 1e-5
    scale = ee_action_norm.clamp(max=max_norm)
    ee_action = ee_action * scale
    action = torch.cat([ee_action, gripper_action], dim=1)
    return action  # noqa: RET504


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6, max_action_norm: float = -1):
        if isinstance(scale, float):
            scale = torch.ones_like(loc) * scale

        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.max_action_norm = max_action_norm

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x  # noqa: RET504

    def sample(self, clip=None, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        x = self._clamp(x)
        if self.max_action_norm > 0:
            x = clip_action_norm(x, self.max_action_norm)
        return x


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
            return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
