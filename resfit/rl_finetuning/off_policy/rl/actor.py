# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout, use_layer_norm=True):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm and layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if use_layer_norm and layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout, use_layer_norm=True):
        super().__init__()

        # if fuse_patch:
        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim

        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        layers = [nn.Linear(proj_in_dim, proj_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(proj_dim))
        layers.append(nn.ReLU(inplace=True))

        self.input_proj = nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z  # noqa: RET504


class Actor(nn.Module):
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: ActorConfig, residual_actor: bool = False):
        super().__init__()

        self.prop_dim = prop_dim
        self.residual_actor = residual_actor
        self.cfg = cfg

        if residual_actor:
            # The residual actor takes the base action as input alongside the state
            self.prop_dim += action_dim

        if cfg.spatial_emb > 0:
            assert cfg.spatial_emb > 1, "this is the dimension"
            self.compress = SpatialEmb(
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                prop_dim=self.prop_dim,
                proj_dim=cfg.spatial_emb,
                dropout=cfg.dropout,
                use_layer_norm=cfg.use_layer_norm,
            )
            policy_in_dim = cfg.spatial_emb
        else:
            layers = [nn.Linear(repr_dim, cfg.feature_dim)]
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(cfg.feature_dim))
            layers.extend([nn.Dropout(cfg.dropout), nn.ReLU()])

            self.compress = nn.Sequential(*layers)
            policy_in_dim = cfg.feature_dim

        if self.prop_dim > 0:
            policy_in_dim += self.prop_dim

        # Create policy network
        self.policy = build_fc(
            policy_in_dim,
            cfg.hidden_dim,
            action_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )

        # Apply weight initialization
        self._initialize_weights(cfg)

    def _initialize_weights(self, cfg: ActorConfig):
        """Apply weight initialization to all networks."""
        # Determine initialization distributions
        intermediate_init = cfg.actor_intermediate_layer_init_distribution
        if cfg.orth and intermediate_init == "default":
            intermediate_init = "orthogonal"

        # Initialize compression layers
        if cfg.orth:
            # Backward compatibility: use existing orthogonal initialization
            self.compress.apply(utils.orth_weight_init)
        else:
            # Use the specified distribution for compression layers
            utils.apply_initialization_to_network(self.compress, intermediate_init)

        # Initialize policy network intermediate layers (exclude final layer)
        utils.apply_initialization_to_network(self.policy, intermediate_init, exclude_final_layer=True)

        # Initialize final layer with specific configuration if provided
        if cfg.actor_last_layer_init_scale is not None:
            final_layer = None
            for module in reversed(list(self.policy.modules())):
                if isinstance(module, nn.Linear):
                    final_layer = module
                    break

            if final_layer is not None:
                utils.initialize_layer_weights(
                    final_layer,
                    cfg.actor_last_layer_init_distribution,
                    cfg.actor_last_layer_init_scale,
                )

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        if isinstance(self.compress, SpatialEmb):
            assert not self.residual_actor, "Not implemented"
            feat = self.compress.forward(obs["feat"], obs["observation.state"])
        else:
            feat = obs["feat"].flatten(1, -1)
            feat = self.compress(feat)

        all_input = [feat]
        if self.prop_dim > 0:
            prop = obs["observation.state"]
            all_input.append(prop)
            if self.residual_actor:
                # The residual actor takes the base action as input alongside the state
                all_input.append(obs["observation.base_action"])

        policy_input = torch.cat(all_input, dim=-1)

        mu: torch.Tensor = self.policy(policy_input)

        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)

        return action_dist  # noqa: RET504
