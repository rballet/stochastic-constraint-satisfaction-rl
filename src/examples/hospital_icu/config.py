"""Configuration module for ICU admission environment."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnvironmentConfig:
    """Environment configuration."""
    capacity: int = 50
    max_rejections: int = 200
    action_space_size: int = 2


@dataclass(frozen=True) 
class RewardConfig:
    """Reward configuration."""
    capacity_weight: float = 0.1
    constraint_penalty: float = 2.0
    success_bonus: float = 10.0
    failure_penalty: float = 5.0
    early_bonus: float = 0.0
    acceptance_reward: float = 0.0
    use_dynamic_scaling: bool = False
    scaling_factor: float = 1.0


def create_config(
    capacity: int = 50,
    max_rejections: int = 200,
    constraint_penalty: float = 2.0,
    success_bonus: float = 10.0,
    **kwargs
) -> tuple[EnvironmentConfig, RewardConfig]:
    """Create environment and reward configurations."""
    env_config = EnvironmentConfig(
        capacity=capacity,
        max_rejections=max_rejections
    )
    
    reward_config = RewardConfig(
        constraint_penalty=constraint_penalty,
        success_bonus=success_bonus,
        **kwargs
    )
    
    return env_config, reward_config