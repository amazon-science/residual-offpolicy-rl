# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from torch import nn


@dataclass
class VitEncoderConfig:
    depth: int = 1
    embed_dim: int = 128
    embed_norm: int = 0
    embed_style: str = "embed2"
    num_heads: int = 4
    patch_size: int = 8
    stride: int = -1
    act_layer = nn.GELU


@dataclass
class CriticLossCfg:
    type: str = "mse"
    n_bins: int = 51  # 0.995 γ → δ=0.005  # noqa: RUF003
    v_min: float = 0.0
    v_max: float = 1.0
    sigma: float = -1.0  # Use -1 to indicate auto-compute: 0.75 * (v_max - v_min) / n_bins

    def __post_init__(self):
        assert self.type in ["mse", "hl_gauss", "c51"]


@dataclass
class CriticConfig:
    drop: float = 0
    feature_dim: int = 128
    fuse_patch: int = 1
    hidden_dim: int = 1024
    norm_weight: int = 0
    orth: int = 1
    spatial_emb: int = 1024

    # Number of independent Q-heads (in the ensemble). Set to 2 for TD3, >2 for RED-Q style.
    num_q: int = 10
    loss: CriticLossCfg = field(default_factory=lambda: CriticLossCfg())
    # Policy gradient type: "ensemble_mean" (mean over all heads, standard RED-Q) or
    # "min_random_pair" (min of 2 random heads) or "q1" (just use q1 from ensemble, standard TD3)
    policy_gradient_type: str = "ensemble_mean"
    # Number of hidden layers in the critic MLP heads (default 2 for backwards compatibility)
    num_layers: int = 2
    # Layer normalization control
    use_layer_norm: bool = True
    # Number of Q-heads to take min over for target computation (default 2 for RED-Q behavior)
    min_q_heads: int = 2

    def __post_init__(self):
        assert self.policy_gradient_type in ["ensemble_mean", "min_random_pair", "q1"], (
            f"Invalid policy_gradient_type: {self.policy_gradient_type}"
        )


@dataclass
class ActorConfig:
    feature_dim: int = 128
    hidden_dim: int = 1024
    dropout: float = 0
    orth: int = 1
    max_action_norm: float = -1
    spatial_emb: int = 0

    # Optional parameter to control last layer initialization scale
    # If None, uses default initialization. If set to a small value (e.g., 1e-3):
    # - For 'normal': used as standard deviation
    # - For 'orthogonal'/'xavier_uniform': used as gain to scale initialization close to zero
    actor_last_layer_init_scale: float | None = None
    # Distribution to use for last layer initialization
    # Options: 'normal', 'orthogonal', 'xavier_uniform'
    actor_last_layer_init_distribution: str = "normal"
    # Distribution to use for intermediate layer initialization
    # Options: 'default', 'normal', 'orthogonal', 'xavier_uniform'
    # 'default' uses PyTorch's default initialization
    actor_intermediate_layer_init_distribution: str = "default"
    # L2 regularization weight on action magnitude
    action_l2_reg_weight: float = 0.0
    action_scale: float = 1.0
    # Number of hidden layers in the actor MLP (default 2 for backwards compatibility)
    num_layers: int = 2
    # Layer normalization control
    use_layer_norm: bool = True


@dataclass
class QAgentConfig:
    device: str = "cuda"
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_clip: float = 0.3
    # LR warmup configuration
    lr_warmup_steps: int = 0  # Number of warmup steps
    lr_warmup_start: float = 1e-8  # Starting LR for warmup (very small positive value)
    # encoder
    use_prop: int = 1
    enc_type: str = "vit"
    vit: VitEncoderConfig = field(default_factory=lambda: VitEncoderConfig())
    # critic & actor
    critic: CriticConfig = field(default_factory=lambda: CriticConfig())
    actor: ActorConfig = field(default_factory=lambda: ActorConfig())

    # gradient clipping
    critic_grad_clip_norm: float = 1.0
    actor_grad_clip_norm: float = 1.0

    # bc loss regularization
    bc_loss_coef: float = 0.0
    bc_loss_dynamic: int = 0  # dynamically scale bc loss weight
    bc_backprop_encoder: bool = False  # Whether BC loss should update encoder (default False for RLPD)

    # encoder freezing
    freeze_encoder: bool = False  # Whether to freeze encoder parameters (no gradient updates)

    clip_q_target_to_reward_range: bool = False

    # TD3 target action noise configuration
    target_action_noise: bool = True  # Whether to add noise to target actions in TD3

    def __post_init__(self):
        pass


# -----------------------------------------------------------------------------
# Algorithm-specific hyper-parameters -----------------------------------------
# -----------------------------------------------------------------------------


@dataclass
class RLPDAlgoConfig:
    """RLPD hyper-parameters."""

    # Training ---------------------------------------------------------------
    total_timesteps: int = 300_000
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 10_000

    # Discounting / target updates ------------------------------------------
    gamma: float = 0.99

    # Update scheduling ------------------------------------------------------
    num_updates_per_iteration: int = 4
    actor_updates_per_iteration: int = 1
    update_every_n_steps: int = 1

    # Offline / online mixture ----------------------------------------------
    offline_fraction: float = 0.5  # fraction of minibatch sampled from offline buffer

    # ------------------------------------------------------------------
    # N-step returns ----------------------------------------------------
    # ------------------------------------------------------------------
    # Horizon used by the MultiStep transform in our replay buffers.
    n_step: int = 3

    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    # Number of critic-only updates before training the actor
    critic_warmup_steps: int = 0

    # ------------------------------------------------------------------
    # Actor learning rate warmup schedule -----------------------------
    # ------------------------------------------------------------------
    # Number of steps over which to linearly warm up the actor learning rate from 0 to full actor_lr
    actor_lr_warmup_steps: int = 0

    # ------------------------------------------------------------------
    # Optional Behavioural Cloning (BC) loss ---------------------------
    # ------------------------------------------------------------------
    # bc_loss_weight: float = 0.0  # Weight for BC loss term in the actor objective

    # ------------------------------------------------------------------
    # Random action exploration -----------------------------------------
    # ------------------------------------------------------------------
    # Scale for random action noise during initial exploration phase
    # Actions are sampled as: rand_actions = torch.rand(...) * 2 * random_action_noise_scale - random_action_noise_scale
    random_action_noise_scale: float = 1.0  # Default: uniform in [-1, 1]

    # ------------------------------------------------------------------
    # Sampling strategy -------------------------------------------------
    # ------------------------------------------------------------------
    # Sampling strategy for replay buffer:
    # "uniform": uniform sampling (alpha=0, beta=0, no priority updates)
    # "prioritized_replay": prioritized experience replay using TD errors
    sampling_strategy: str = "uniform"

    # Prefetch batches using background threads to speed up sampling
    prefetch_batches: int = 4  # Number of batches to prefetch (0 = disabled, 4-8 recommended)

    # Priority parameters for prioritized experience replay
    priority_alpha: float = 0.6  # Controls prioritization strength (0=uniform, higher=more prioritization)
    priority_beta: float = 0.4  # Controls importance sampling correction

    # ------------------------------------------------------------------
    # Standard deviation schedule -------------------------------------------
    # ------------------------------------------------------------------
    stddev_max: float = 0.1
    stddev_min: float = 0.1
    stddev_step: int = 300_000

    stddev_schedule: str = field(init=False)

    def __post_init__(self):
        self.stddev_schedule = f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


# -----------------------------------------------------------------------------
# Dataset specification --------------------------------------------------------
# -----------------------------------------------------------------------------


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: int | None = 300
    image_key: str | None = None


@dataclass
class WandBConfig:
    project: str = "rlpd"
    name: str | None = None
    mode: str = "online"
    entity: str | None = None
    notes: str | None = None
    continue_run_id: str | None = None
    group: str | None = None


# -----------------------------------------------------------------------------
# Top-level experiment config --------------------------------------------------
# -----------------------------------------------------------------------------
@dataclass
class RLPDDexmgConfig:
    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    seed: int | None = None
    torch_deterministic: bool = False
    debug: bool = False

    # ------------------------------------------------------------------
    # Task / environment
    # ------------------------------------------------------------------
    task: str = "Can"
    num_envs: int = 1
    eval_num_envs: int = 8
    eval_num_episodes: int = 50
    headless: bool = True
    video_key: str = "observation.images.agentview"
    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ]
    )

    # ------------------------------------------------------------------
    # Algorithm & optimisation
    # ------------------------------------------------------------------
    algo: RLPDAlgoConfig = field(default_factory=RLPDAlgoConfig)

    # ------------------------------------------------------------------
    # Network architectures
    # ------------------------------------------------------------------
    agent: QAgentConfig = field(default_factory=QAgentConfig)

    # ------------------------------------------------------------------
    # Offline dataset
    # ------------------------------------------------------------------
    offline_data: OfflineDataConfig | None = field(default_factory=OfflineDataConfig)

    # ------------------------------------------------------------------
    # Weights & Biases logging
    # ------------------------------------------------------------------
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    log_freq: int = 100
    eval_interval_every_steps: int = 10_000
    checkpoint_interval: int = -1
    save_video: bool = True
    # Whether to run an evaluation pass before training begins (at step 0)
    eval_first: bool = True


@dataclass
class RLPDCanConfig(RLPDDexmgConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
        )
    )

    # NOTE: We use the `ibrl` suffix to differentiate from the original bad RLPD implementation,
    # should be renamed to `rlpd` eventually
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-can-rlpd-ibrl"))


@dataclass
class RLPDSquareConfig(RLPDDexmgConfig):
    task: str = "Square"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-square-rlpd"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
        )
    )


@dataclass
class RLPDBoxCleanConfig(RLPDDexmgConfig):
    task: str = "TwoArmBoxCleanup"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-box-clean-rlpd"))

    # Use same camera setup as residual TD3 BoxCleanup config
    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-box-cleanup",
            num_episodes=1_000,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            total_timesteps=500_000,
        )
    )


@dataclass
class RLPDCoffeeConfig(RLPDDexmgConfig):
    task: str = "TwoArmCoffee"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-coffee-rlpd"))

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-coffee",
            num_episodes=1_000,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            total_timesteps=500_000,
        )
    )


@dataclass
class RLPDThreadingConfig(RLPDDexmgConfig):
    task: str = "Threading"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="mimicgen-threading-rlpd"))

    # Use single-arm camera setup for MimicGen threading task
    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/mimicgen-d0-threading",
            num_episodes=300,
        )
    )

    # Threading is a precise manipulation task that may need more training
    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            gamma=0.997,
        )
    )


@dataclass
class RLPDTwoArmThreadingConfig(RLPDDexmgConfig):
    task: str = "TwoArmThreading"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-threading-rlpd"))

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-threading",
            num_episodes=1_000,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            total_timesteps=500_000,
        )
    )


@dataclass
class RLPDPouringConfig(RLPDDexmgConfig):
    task: str = "TwoArmPouring"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-pouring-rlpd"))

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-pouring",
            num_episodes=1_000,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            total_timesteps=500_000,
        )
    )


@dataclass
class RLPDLiftTrayConfig(RLPDDexmgConfig):
    task: str = "TwoArmLiftTray"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-lifttray-rlpd"))

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-lift-tray",
            num_episodes=1_000,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            total_timesteps=500_000,
        )
    )


@dataclass
class RLPDThreePieceAssemblyConfig(RLPDDexmgConfig):
    task: str = "TwoArmThreePieceAssembly"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-threepieceassembly-rlpd"))

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-three-piece-assembly",
            num_episodes=1_000,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            total_timesteps=500_000,
        )
    )


@dataclass
class RLPDTwoArmTransportConfig(RLPDDexmgConfig):
    task: str = "TwoArmTransport"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-transport-rlpd"))

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.shouldercamera0",
            "observation.images.shouldercamera1",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-transport-image",
            num_episodes=300,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            gamma=0.998,
            total_timesteps=500_000,
        )
    )


@dataclass
class RLPDTwoArmCanSortConfig(RLPDDexmgConfig):
    task: str = "TwoArmCanSortRandom"

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-cansort-rlpd"))

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.frontview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-can-sort-random",
            num_episodes=1_000,
        )
    )

    algo: RLPDAlgoConfig = field(
        default_factory=lambda: RLPDAlgoConfig(
            total_timesteps=500_000,
        )
    )


# -----------------------------------------------------------------------------
# Register with Hydra
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(name="rlpd_dexmg_config", node=RLPDDexmgConfig)
cs.store(name="rlpd_can_config", node=RLPDCanConfig)
cs.store(name="rlpd_square_config", node=RLPDSquareConfig)
cs.store(name="rlpd_box_clean_config", node=RLPDBoxCleanConfig)
cs.store(name="rlpd_coffee_config", node=RLPDCoffeeConfig)
cs.store(name="rlpd_threading_config", node=RLPDThreadingConfig)
cs.store(name="rlpd_two_arm_threading_config", node=RLPDTwoArmThreadingConfig)
cs.store(name="rlpd_pouring_config", node=RLPDPouringConfig)
cs.store(name="rlpd_lift_tray_config", node=RLPDLiftTrayConfig)
cs.store(name="rlpd_three_piece_assembly_config", node=RLPDThreePieceAssemblyConfig)
cs.store(name="rlpd_two_arm_transport_config", node=RLPDTwoArmTransportConfig)
cs.store(name="rlpd_two_arm_cansort_config", node=RLPDTwoArmCanSortConfig)
