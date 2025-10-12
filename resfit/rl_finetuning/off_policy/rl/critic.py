# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import math

import torch
from torch import nn

# Import vmap functionality (PyTorch 2.1+)
from torch.func import functional_call, stack_module_state, vmap
from torch.nn import functional

from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float | None = None):
        super().__init__()
        if sigma is None:
            # Use the recommended sigma from the paper
            sigma = 0.75 * (max_value - min_value) / num_bins

        self.sigma = sigma
        # Create bin edges (num_bins + 1 points)
        bin_edges = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer("bin_edges", bin_edges)

        # Bin centers for expectation computation
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer("bin_centers", bin_centers)

    def _log1mexp(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(1 - exp(-|x|)) in numerically stable way."""
        x = torch.abs(x)
        return torch.where(x < math.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

    def _log_sub_exp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log(exp(max(x,y)) - exp(min(x,y))) in numerically stable way."""
        larger = torch.maximum(x, y)
        smaller = torch.minimum(x, y)
        return larger + self._log1mexp(torch.maximum(larger - smaller, torch.zeros_like(larger)))

    def _log_ndtr(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(Φ(x)) where Φ is standard normal CDF, numerically stable."""
        # For x > 6, use asymptotic expansion
        # For x < -6, use log(Φ(x)) ≈ -x²/2 - log(√(2π)) - log(-x)
        # For -6 ≤ x ≤ 6, use log(0.5 * (1 + erf(x/√2)))

        ndtr_vals = torch.zeros_like(x)

        # For large positive x (> 6): log(Φ(x)) ≈ log(1 - Φ(-x)) ≈ log(1 - e^(-x²/2 - log(√(2π)) + log(-x)))
        large_pos = x > 6
        if torch.any(large_pos):
            ndtr_vals[large_pos] = torch.log(torch.special.ndtr(x[large_pos]))

        # For large negative x (< -6): log(Φ(x)) ≈ -x²/2 - log(√(2π)) - log(-x)
        large_neg = x < -6
        if torch.any(large_neg):
            x_neg = x[large_neg]
            ndtr_vals[large_neg] = -x_neg * x_neg / 2 - math.log(math.sqrt(2 * math.pi)) - torch.log(-x_neg)

        # For moderate x (-6 ≤ x ≤ 6): use direct computation
        moderate = (~large_pos) & (~large_neg)
        if torch.any(moderate):
            x_mod = x[moderate]
            # Use erfc for better precision when x is negative
            erfc_val = torch.special.erfc(-x_mod / math.sqrt(2))
            ndtr_vals[moderate] = torch.log(0.5 * erfc_val)

        return ndtr_vals

    def _normal_cdf_log_difference(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log(Φ(x) - Φ(y)) where Φ is standard normal CDF."""
        # When x >= y >= 0, compute log(Φ(-y) - Φ(-x)) for better precision
        is_y_positive = y >= 0
        x_hat = torch.where(is_y_positive, -y, x)
        y_hat = torch.where(is_y_positive, -x, y)

        log_ndtr_x = self._log_ndtr(x_hat)
        log_ndtr_y = self._log_ndtr(y_hat)

        return self._log_sub_exp(log_ndtr_x, log_ndtr_y)

    def _target_to_probs(self, target):
        target = target.unsqueeze(-1)  # [batch] -> [batch, 1]

        # Compute normalized distances to bin edges
        norm_factor = math.sqrt(2) * self.sigma
        upper_edges = (self.bin_edges[1:].unsqueeze(0) - target) / norm_factor
        lower_edges = (self.bin_edges[:-1].unsqueeze(0) - target) / norm_factor

        # Compute log probabilities for each bin
        bin_log_probs = self._normal_cdf_log_difference(upper_edges, lower_edges)

        # Compute normalization factor (log of total probability mass)
        log_z = self._normal_cdf_log_difference(
            (self.bin_edges[-1] - target) / norm_factor, (self.bin_edges[0] - target) / norm_factor
        )

        # Convert to probabilities
        probs = torch.exp(bin_log_probs - log_z.unsqueeze(-1))

        # Safety check: if anything goes wrong, fall back to uniform
        valid_mask = torch.isfinite(probs).all(dim=-1, keepdim=True)
        uniform_probs = torch.ones_like(probs) / probs.shape[-1]
        return torch.where(valid_mask, probs, uniform_probs)

    def forward(self, logits, target):
        tgt_probs = self._target_to_probs(target.detach())
        # Manual cross-entropy for soft labels: -sum(p * log_softmax(logits))
        log_probs = functional.log_softmax(logits, dim=-1)
        return -(tgt_probs * log_probs).sum(dim=-1).mean()

    def forward_batched(self, logits_batch, target):
        """
        Compute HLGaussLoss for multiple critic heads in a batched manner.
        This exactly replicates the original (buggy) behavior but in a vectorized way.

        Args:
            logits_batch: [num_heads, batch_size, num_bins] - logits from all critic heads
            target: [batch_size] - target values

        Returns:
            loss: scalar - average loss across all heads and batch
        """
        num_heads = logits_batch.shape[0]
        batch_size = target.shape[0]

        # Get the buggy target probabilities (same for all heads)
        tgt_probs_buggy = self._target_to_probs(target.detach())  # [batch_size, batch_size, num_bins]

        # Compute log softmax for all heads
        log_probs = functional.log_softmax(logits_batch, dim=-1)  # [num_heads, batch_size, num_bins]

        # Expand tgt_probs_buggy to all heads: [B, B, bins] -> [heads, B, B, bins]
        tgt_probs_expanded = tgt_probs_buggy.unsqueeze(0).expand(num_heads, -1, -1, -1)

        # Expand log_probs to match: [num_heads, batch_size, num_bins] -> [num_heads, batch_size, batch_size, num_bins]
        log_probs_expanded = log_probs.unsqueeze(2).expand(-1, -1, batch_size, -1)

        # Multiply: [num_heads, batch_size, batch_size, num_bins]
        product = tgt_probs_expanded * log_probs_expanded

        # Sum over bins: [num_heads, batch_size, batch_size]
        summed = product.sum(dim=-1)

        # Mean over all dimensions except heads, then mean over heads
        # This matches the original: -summed_h.mean() for each head, then average across heads
        losses_per_head = -summed.view(num_heads, -1).mean(dim=-1)  # [num_heads]

        return losses_per_head.mean()


class C51Loss(nn.Module):
    def __init__(self, v_min: float, v_max: float, num_atoms: int):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

        # Create support for value distribution
        support = torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        # Delta z for projection
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def project_distribution(
        self, next_distribution: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Project Bellman update onto categorical support."""
        batch_size = rewards.size(0)

        # Compute target values for each atom: r + gamma * (1 - done) * support
        target_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)

        # Clamp target support to valid range
        target_support = torch.clamp(target_support, self.v_min, self.v_max)

        # Compute indices and interpolation weights for projection
        b = (target_support - self.v_min) / self.delta_z
        lower = b.floor().long()  # Lower bound indices
        upper = b.ceil().long()  # Upper bound indices

        # Handle edge cases
        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.num_atoms - 1)) * (lower == upper)] += 1

        # Project probabilities onto support
        target_distribution = torch.zeros_like(next_distribution)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, dtype=torch.long, device=target_distribution.device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
        )

        # Lower bound projection
        target_distribution.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_distribution * (upper.float() - b)).view(-1)
        )
        # Upper bound projection
        target_distribution.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_distribution * (b - lower.float())).view(-1)
        )

        return target_distribution

    def forward(self, current_logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """Compute C51 loss using cross-entropy between current and target distributions."""
        # Convert logits to log probabilities
        current_log_probs = functional.log_softmax(current_logits, dim=-1)

        # Compute cross-entropy loss
        return -(target_distribution * current_log_probs).sum(dim=-1).mean()

    def logits_to_q_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert categorical logits to Q-value using expectation over support."""
        probs = functional.softmax(logits, dim=-1)
        return (probs * self.support).sum(dim=-1, keepdim=True)

    def forward_batched(self, current_logits_batch: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """
        Compute C51 loss for multiple critic heads in a batched manner.

        Args:
            current_logits_batch: [num_heads, batch_size, num_atoms] - logits from all critic heads
            target_distribution: [batch_size, num_atoms] - target distribution (same for all heads)

        Returns:
            loss: scalar - average loss across all heads and batch
        """
        # Convert logits to log probabilities for all heads
        current_log_probs = functional.log_softmax(current_logits_batch, dim=-1)

        # Expand target distribution to match number of heads
        num_heads = current_logits_batch.shape[0]
        target_expanded = target_distribution.unsqueeze(0).expand(num_heads, -1, -1)

        # Compute cross-entropy loss for all heads at once
        losses = -(target_expanded * current_log_probs).sum(dim=-1)  # [num_heads, batch_size]
        return losses.mean()  # Average over all heads and batch


class Critic(nn.Module):
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = cfg.loss

        if self.loss_cfg.type in {"hl_gauss", "c51"}:
            output_dim = self.loss_cfg.n_bins
        else:
            output_dim = 1

        # Number of Q-heads (ensemble size)
        num_q = getattr(cfg, "num_q", 2)

        # Build an ensemble that shares the spatial trunk and owns K independent heads
        self.q_ensemble = SpatialEmbQEnsemble(
            fuse_patch=cfg.fuse_patch,
            num_patch=repr_dim // patch_repr_dim,
            patch_dim=patch_repr_dim,
            emb_dim=cfg.spatial_emb,
            prop_dim=prop_dim,
            action_dim=action_dim,
            hidden_dim=self.cfg.hidden_dim,
            orth=self.cfg.orth,
            output_dim=output_dim,
            num_heads=num_q,
            num_layers=cfg.num_layers,
            use_layer_norm=cfg.use_layer_norm,
        )

        # Loss objects (single instance) ----
        if self.loss_cfg.type == "hl_gauss":
            # Use None if sigma is -1.0 (auto-compute), otherwise use the specified value
            sigma_val = None if self.loss_cfg.sigma < 0 else self.loss_cfg.sigma
            self.hl_loss = HLGaussLoss(
                min_value=self.loss_cfg.v_min,
                max_value=self.loss_cfg.v_max,
                num_bins=self.loss_cfg.n_bins,
                sigma=sigma_val,
            )
        elif self.loss_cfg.type == "c51":
            self.c51_loss = C51Loss(
                v_min=self.loss_cfg.v_min,
                v_max=self.loss_cfg.v_max,
                num_atoms=self.loss_cfg.n_bins,
            )

    @staticmethod
    def _logits_to_q(probs, support):
        # (B,K) * (K,) → (B,)
        return (probs * support).sum(-1, keepdim=True)  # E[z]

    def forward(self, feat, prop, act, *, return_logits: bool = False):
        # logits_per_head: [num_q, B, out_dim] where out_dim is either 1 or K (bins)
        logits_per_head = self.q_ensemble(feat, prop, act)

        if self.loss_cfg.type == "hl_gauss":
            # expectation over bin centers → scalar Q, done per head
            q_per_head = self._logits_to_q(
                torch.softmax(logits_per_head, dim=-1),
                self.hl_loss.bin_centers,
            )  # [num_q, B, 1]

            if return_logits:
                return q_per_head, logits_per_head
            return q_per_head

        if self.loss_cfg.type == "c51":
            # expectation over categorical support → scalar Q, done per head
            q_per_head = self.c51_loss.logits_to_q_value(logits_per_head)  # [num_q, B, 1]

            if return_logits:
                return q_per_head, logits_per_head
            return q_per_head

        # MSE case: outputs are already scalars per head → [num_q, B, 1]
        return logits_per_head

    def q_value(self, feat, prop, act):
        """
        Returns the Q-value for a given feature, property, and action.
        I.e., gets all Q-values, subsets them, and returns the min.
        Used for critic loss computation (uses configurable min_q_heads).
        """
        # Get the Q-values for all heads
        q_out = self.forward(feat, prop, act)

        # Take min over random subset of heads (configurable via min_q_heads)
        num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
        idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
        return torch.min(q_out.index_select(0, idx), dim=0).values

    def q_value_for_policy(self, feat, prop, act):
        """
        Returns the Q-value for policy gradient computation.
        Supports different policy gradient types:
        - "ensemble_mean": mean over all heads (standard RED-Q approach)
        - "min_random_pair": min of configurable number of random heads (conservative variant)
        - "q1": just use q1 from ensemble (standard TD3 approach)
        """
        # Get the Q-values for all heads
        q_out = self.forward(feat, prop, act)

        if self.cfg.policy_gradient_type == "ensemble_mean":
            # Take mean over all heads (standard RED-Q approach)
            return q_out.mean(dim=0)
        if self.cfg.policy_gradient_type == "min_random_pair":
            # Take min over random subset of heads (configurable via min_q_heads)
            num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
            idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
            return torch.min(q_out.index_select(0, idx), dim=0).values
        if self.cfg.policy_gradient_type == "q1":
            return q_out[0]
        raise ValueError(f"Unknown policy_gradient_type: {self.cfg.policy_gradient_type}")


class SpatialEmbQNet(nn.Module):
    def __init__(
        self,
        num_patch,
        patch_dim,
        prop_dim,
        action_dim,
        fuse_patch,
        emb_dim,
        hidden_dim,
        orth,
        output_dim=1,
        use_layer_norm=True,
    ):
        super().__init__()

        if fuse_patch:
            proj_in_dim = num_patch + action_dim + prop_dim
            num_proj = patch_dim
        else:
            proj_in_dim = patch_dim + action_dim + prop_dim
            num_proj = num_patch

        self.fuse_patch = fuse_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        # Build input projection layers
        input_layers = [nn.Linear(proj_in_dim, emb_dim)]
        if use_layer_norm:
            input_layers.append(nn.LayerNorm(emb_dim))
        input_layers.append(nn.ReLU(inplace=True))
        self.input_proj = nn.Sequential(*input_layers)

        self.weight = nn.Parameter(torch.zeros(1, num_proj, emb_dim))
        nn.init.normal_(self.weight)

        # Build Q network layers
        q_layers = [nn.Linear(emb_dim + action_dim + prop_dim, hidden_dim)]
        if use_layer_norm:
            q_layers.append(nn.LayerNorm(hidden_dim))
        q_layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)])
        if use_layer_norm:
            q_layers.append(nn.LayerNorm(hidden_dim))
        q_layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim)])
        self.q = nn.Sequential(*q_layers)
        if orth:
            self.q.apply(utils.orth_weight_init)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor):
        assert feat.size(-1) == self.patch_dim, "are you using CNN, need flatten&transpose"

        if self.fuse_patch:
            feat = feat.transpose(1, 2)

        repeated_action = action.unsqueeze(1).repeat(1, feat.size(1), 1)
        all_feats = [feat, repeated_action]
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            all_feats.append(repeated_prop)

        x = torch.cat(all_feats, dim=-1)
        y: torch.Tensor = self.input_proj(x)
        z = (self.weight * y).sum(1)

        if self.prop_dim == 0:
            z = torch.cat((z, action), dim=-1)
        else:
            z = torch.cat((z, prop, action), dim=-1)

        q = self.q(z)
        # For MSE: [batch, 1], For HL-Gauss: [batch, K]
        return q  # noqa: RET504


class HeadMLP(nn.Module):
    """Single MLP head for the ensemble."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, use_layer_norm: bool = True):
        super().__init__()

        # Build layers dynamically based on num_layers
        layers = []
        current_dim = in_dim

        # Add hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())  # avoid inplace with vmap
            current_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(current_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            z: Input tensor [batch_size, in_dim]

        Returns:
            Output tensor [batch_size, out_dim]
        """
        return self.net(z)


class SpatialEmbQEnsemble(nn.Module):
    def __init__(
        self,
        *,
        num_patch: int,
        patch_dim: int,
        prop_dim: int,
        action_dim: int,
        fuse_patch: int,
        emb_dim: int,
        hidden_dim: int,
        orth: int,
        output_dim: int = 1,
        num_heads: int = 2,
        num_layers: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        # Trunk (shared across heads)
        if fuse_patch:
            proj_in_dim = num_patch + action_dim + prop_dim
            num_proj = patch_dim
        else:
            proj_in_dim = patch_dim + action_dim + prop_dim
            num_proj = num_patch

        self.fuse_patch = fuse_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim
        self.action_dim = action_dim

        # Build input projection layers
        input_layers = [nn.Linear(proj_in_dim, emb_dim)]
        if use_layer_norm:
            input_layers.append(nn.LayerNorm(emb_dim))
        input_layers.append(nn.ReLU(inplace=True))
        self.input_proj = nn.Sequential(*input_layers)
        self.weight = nn.Parameter(torch.zeros(1, num_proj, emb_dim))
        nn.init.normal_(self.weight)

        # vmap-based heads for efficient batched computation
        self.num_heads = num_heads
        input_dim = emb_dim + action_dim + prop_dim

        # Create multiple module instances and stack their parameters/buffers
        heads = [HeadMLP(input_dim, hidden_dim, output_dim, num_layers, use_layer_norm) for _ in range(num_heads)]
        self.params, self.buffers = stack_module_state(heads)

        # Store a template head with no parameters (for structure only)
        self._head_template = HeadMLP(input_dim, hidden_dim, output_dim, num_layers, use_layer_norm)

        # Register params and buffers so they get moved with the module
        for name, param in self.params.items():
            self.register_parameter(f"_vmap_param_{name.replace('.', '_')}", nn.Parameter(param))
        for name, buffer in self.buffers.items():
            self.register_buffer(f"_vmap_buffer_{name.replace('.', '_')}", buffer)

        # Apply per-head initialization
        self._init_per_head_params(orth)

    def _init_per_head_params(self, orth: bool):
        """Apply per-head initialization to the stacked parameters."""
        with torch.no_grad():
            if orth:
                # Apply orthogonal initialization to all linear layers
                for key, param in self.params.items():
                    if "weight" in key and param.dim() >= 2:
                        # param shape: [num_heads, out_features, in_features] or similar
                        for h in range(self.num_heads):
                            if param[h].dim() >= 2:
                                utils.orth_weight_init(param[h])

    def extra_repr(self) -> str:
        return f"heads: {self.num_heads}, weight: nn.Parameter ({self.weight.size()})"

    def _compute_trunk(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert feat.size(-1) == self.patch_dim, "are you using CNN, need flatten&transpose"
        if self.fuse_patch:
            feat = feat.transpose(1, 2)

        repeated_action = action.unsqueeze(1).repeat(1, feat.size(1), 1)
        all_feats = [feat, repeated_action]
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            all_feats.append(repeated_prop)

        x = torch.cat(all_feats, dim=-1)
        y: torch.Tensor = self.input_proj(x)
        z = (self.weight * y).sum(1)

        if self.prop_dim == 0:
            z = torch.cat((z, action), dim=-1)
        else:
            z = torch.cat((z, prop, action), dim=-1)
        return z

    def forward(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self._compute_trunk(feat, prop, action)  # [batch_size, input_dim]

        # Reconstruct params/buffers from registered tensors (ensures proper device placement)
        current_params = {}
        current_buffers = {}

        for name in self.params:
            param_name = f"_vmap_param_{name.replace('.', '_')}"
            current_params[name] = getattr(self, param_name)

        for name in self.buffers:
            buffer_name = f"_vmap_buffer_{name.replace('.', '_')}"
            current_buffers[name] = getattr(self, buffer_name)

        def f_one_head(p, b, z_input):
            # p, b are the param/buffer for one head; z_input is [B, in_dim]
            return functional_call(self._head_template, (p, b), (z_input,))  # [B, out_dim]

        # Vectorize across heads; params/buffers carry leading H; z is shared across heads
        # Output: [num_heads, batch_size, output_dim]
        return vmap(f_one_head, in_dims=(0, 0, None))(current_params, current_buffers, z)
