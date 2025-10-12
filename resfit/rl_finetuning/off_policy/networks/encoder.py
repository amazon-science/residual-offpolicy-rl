# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import VitEncoderConfig
from resfit.rl_finetuning.off_policy.networks.min_vit import MinVit


class VitEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], cfg: VitEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def forward(self, obs, flatten=True) -> torch.Tensor:
        if obs.max() > 5:
            obs = obs / 255.0
        obs = obs - 0.5
        feats: torch.Tensor = self.vit.forward(obs)
        if flatten:
            # [B, D, N] -> [B, D*N]
            feats = feats.flatten(1, 2)
        return feats


if __name__ == "__main__":
    vit_encoder = VitEncoder(obs_shape=(3, 84, 84), cfg=VitEncoderConfig())
    obs = torch.rand(10, 3, 84, 84)
    feats: torch.Tensor = vit_encoder(obs)
    print(feats.size())  # (10, 10368), i.e., 81 patches * 128 dimensions
