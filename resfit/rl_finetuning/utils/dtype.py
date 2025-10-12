# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import numpy as np
import torch


def to_uint8(obs_dict: dict, keys: list[str]):
    """In-place conversion of image entries in *obs_dict* to uint8.

    Assumes images are float32 in [0,1].  Non-image entries are left
    untouched.  Supports torch.Tensor and numpy.ndarray inputs.
    """

    for _k in keys:
        if _k not in obs_dict:
            continue
        _v = obs_dict[_k]
        if isinstance(_v, torch.Tensor):
            if _v.dtype == torch.uint8:
                continue  # already uint8
            # Avoid inplace on shared tensors
            obs_dict[_k] = (_v * 255.0).clamp_(0, 255).to(torch.uint8)
        elif isinstance(_v, np.ndarray):
            if _v.dtype == np.uint8:
                continue
            obs_dict[_k] = (_v * 255.0).clip(0, 255).astype(np.uint8)
    return obs_dict
