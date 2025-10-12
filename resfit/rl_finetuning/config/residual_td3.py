# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig, RLPDAlgoConfig, RLPDDexmgConfig


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: int | None = 300
    # Offline data action labeling options
    use_base_policy_for_base_actions: bool = True
    # Normalization safeguards
    min_action_range: float = 1e-1  # Minimum range for any action dimension to prevent normalization blow-up
    min_state_std: float = 1e-1  # Minimum std for any state dimension to prevent normalization blow-up


@dataclass
class WandBConfig:
    project: str = "robomimic-can-residual-td3"
    mode: str = "online"
    entity: str | None = None
    notes: str | None = None
    continue_run_id: str | None = None
    name: str | None = None
    group: str | None = None


@dataclass
class BasePolicyConfig:
    wandb_id: str = "robomimic-can-bc/sdo8cku7"
    wt_type: str = "best"
    wt_version: str = "latest"


@dataclass
class ResidualTD3AlgoConfig(RLPDAlgoConfig):
    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    # Number of critic-only updates before training the actor
    critic_warmup_steps: int = 10_000

    # ------------------------------------------------------------------
    # Random action exploration -----------------------------------------
    # ------------------------------------------------------------------
    # Scale for random action noise during initial exploration phase
    # Actions are sampled as: rand_actions = torch.rand(...) * 2 * random_action_noise_scale - random_action_noise_scale
    random_action_noise_scale: float = 0.2  # Default: uniform in [-1, 1]

    # Whether to use base policy + noise (True) or pure uniform noise (False) during warmup
    # Note: Environment wrapper always applies base_action + residual_action
    # True: residual_action = noise (resulting in base_action + noise)
    # False: residual_action = pure_random - base_action (resulting in pure_random)
    use_base_policy_for_warmup: bool = True

    # ------------------------------------------------------------------
    # Standard deviation schedule -------------------------------------------
    # ------------------------------------------------------------------
    stddev_max: float = 0.05
    stddev_min: float = 0.05
    stddev_step: int = 300_000

    # Progressive clipping schedule for the residual actions
    # I.e., starts clipping linearly from 0 to action scale over progressive_clipping_steps steps
    progressive_clipping_steps: int = 0


# -----------------------------------------------------------------------------
# Top-level experiment config --------------------------------------------------
# -----------------------------------------------------------------------------
@dataclass
class ResidualTD3DexmgConfig(RLPDDexmgConfig):
    actor_name: str | None = None  # Inferred from base policy config

    # ------------------------------------------------------------------
    # Algorithm & optimisation
    # ------------------------------------------------------------------
    algo: ResidualTD3AlgoConfig = field(default_factory=ResidualTD3AlgoConfig)

    # ------------------------------------------------------------------
    # Network architectures
    # ------------------------------------------------------------------
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,  # imp for residual
            ),
        )
    )

    # ------------------------------------------------------------------
    # Offline dataset
    # ------------------------------------------------------------------
    offline_data: OfflineDataConfig | None = field(default_factory=OfflineDataConfig)

    # ------------------------------------------------------------------
    # Base policy
    # ------------------------------------------------------------------
    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)

    # ------------------------------------------------------------------
    # Weights & Biases logging
    # ------------------------------------------------------------------
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    eval_interval_every_steps: int = 10_000

    # Whether to run an evaluation pass before training begins (at step 0)
    eval_first: bool = True


@dataclass
class ResidualTD3CanConfig(ResidualTD3DexmgConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-can-bc/sdo8cku7",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-can-residual-td3"))


@dataclass
class ResidualTD3SquareConfig(ResidualTD3DexmgConfig):
    task: str = "Square"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-square-bc/dzbkdpwp",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-square-residual-td3"))


@dataclass
class ResidualTD3BoxCleanConfig(ResidualTD3DexmgConfig):
    task: str = "TwoArmBoxCleanup"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-box-clean-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-box-cleanup",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


@dataclass
class ResidualTD3CoffeeConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCoffee"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="dexmg-coffee-residual-td3", notes="all cameras")
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-coffee",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )

@dataclass
class ResidualTD3TwoArmCanSortConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCanSortRandom"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.frontview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-cansort-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-can-sort-random",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


# -----------------------------------------------------------------------------
# Register with Hydra
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(name="residual_td3_dexmg_config", node=ResidualTD3DexmgConfig)
cs.store(name="residual_td3_can_config", node=ResidualTD3CanConfig)
cs.store(name="residual_td3_square_config", node=ResidualTD3SquareConfig)
cs.store(name="residual_td3_box_clean_config", node=ResidualTD3BoxCleanConfig)
cs.store(name="residual_td3_coffee_config", node=ResidualTD3CoffeeConfig)
cs.store(name="residual_td3_two_arm_cansort_config", node=ResidualTD3TwoArmCanSortConfig)
