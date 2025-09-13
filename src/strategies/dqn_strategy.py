"""
Deep Q-Network (DQN) strategy implementation using Stable-Baselines3 for Stochastic Constraint Satisfaction Problems.

This module provides a DQN implementation built on Stable-Baselines3 including:
- Gymnasium environment wrapper for SCSP
- Stable-Baselines3 DQN agent with advanced features
- Custom reward engineering for constraint satisfaction
- Integration with SCSP framework

Using SB3 provides access to optimized implementations, advanced techniques like prioritized replay,
dueling networks, and comprehensive logging/monitoring capabilities.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import tempfile
import os

# Stable-Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from ..core.strategy_base import AbstractStrategy
from ..core.types import Entity, ProblemState, Decision, Scenario, Constraint
from ..simulation.entity_generator import create_entity_generator


@dataclass
class DQNConfig:
    """Configuration for Stable-Baselines3 DQN strategy."""

    model_path: Optional[str] = None
    
    # SB3 DQN parameters
    learning_rate: float = 0.0005
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    
    # Exploration parameters
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    
    # Network architecture
    net_arch: List[int] = field(default_factory=lambda: [128, 64])
    
    # Advanced DQN features
    optimize_memory_usage: bool = False
    # NOTE: SB3's standard DQN doesn't support prioritized replay
    # For prioritized replay, use Rainbow DQN or implement custom buffer
    
    # SCSP-specific reward parameters
    capacity_reward_weight: float = 0.1
    constraint_reward_weight: float = 2.0
    terminal_reward_scale: float = 10.0
    dilution_penalty_weight: float = 0.05
    
    # Training configuration
    total_timesteps: int = 100000
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    
    # State space configuration (replaces magic numbers)
    max_normalized_value: float = 2.0  # Cap for normalized percentages
    max_urgency_score: float = 5.0     # Cap for urgency scores
    observation_low: float = 0.0       # Lower bound for observation space
    observation_high: float = 5.0      # Upper bound for observation space
    reward_clip_min: float = -10.0     # Minimum reward value
    reward_clip_max: float = 10.0      # Maximum reward value
    
    # Dynamic calculation parameters
    remaining_entities_multiplier: float = 2.0  # For estimating remaining entities
    min_remaining_entities: int = 10            # Minimum estimate for remaining entities
    constraint_satisfaction_threshold_multiplier: float = 1.1  # 110% of requirement for dilution checks
    low_capacity_threshold: float = 0.5        # Threshold for low capacity penalty
    
    # Progress and urgency calculation
    progress_bonus_scale: float = 2.0           # Bonus for satisfying constraints
    base_rejection_penalty: float = 0.1        # Base penalty for rejections
    urgency_scaling_factor: float = 2.0        # Multiplier for urgency-based rewards
    dilution_urgency_threshold: float = 1.0    # Urgency threshold for dilution penalties
    dilution_maintenance_penalty: float = 0.2  # Penalty for diluting satisfied constraints
    critical_urgency_scaling: float = 0.5      # Scaling for critical constraint rejections
    
    # Logging and saving
    verbose: int = 1
    seed: Optional[int] = None


class SCSPEnvironment(gym.Env):
    """Gymnasium environment wrapper for Stochastic Constraint Satisfaction Problems."""
    
    def __init__(self, scenario: Scenario, config: DQNConfig, entity_generator_type: str = 'multivariate'):
        super().__init__()
        
        self.scenario = scenario
        self.config = config
        self.entity_generator = create_entity_generator(entity_generator_type, seed=config.seed)
        
        # Use scenario's attributes as the fixed attribute space
        # For true genericity, this should be the same across all scenarios in a domain
        self.fixed_attributes = scenario.attributes
        
        # Calculate FIXED state dimension (same for ALL scenarios)
        self.state_dim = self._calculate_fixed_state_dim()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0=REJECT, 1=ACCEPT
        # Some observations can exceed 1.0 (e.g., normalized percentages, urgency scores)
        self.observation_space = spaces.Box(
            low=config.observation_low, high=config.observation_high, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Episode state
        self.problem_state = None
        self.current_entity = None
        self.entity_count = 0
        self.max_entities = scenario.capacity + scenario.max_rejections
        
    def _calculate_fixed_state_dim(self) -> int:
        """Calculate FIXED state dimension based on positional encoding for ALL attributes.
        
        This creates a consistent state space where each attribute has a fixed position,
        regardless of whether it's constrained in the current scenario.
        """
        num_fixed_attributes = len(self.fixed_attributes)
        
        # State components breakdown (FIXED for all scenarios):
        basic_state = 4                                    # capacity_util, reject_rate, remaining_capacity, progress_ratio
        attribute_constraint_features = num_fixed_attributes * 4  # per-attribute: percentage, deficit, urgency, satisfaction
        entity_features = num_fixed_attributes             # current entity attributes (binary)
        temporal_features = 2                              # entities_processed_ratio, estimated_entities_remaining_ratio
        
        total_dim = basic_state + attribute_constraint_features + entity_features + temporal_features
        return total_dim
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.entity_generator.reset(seed)
        else:
            self.entity_generator.reset()
        
        # Initialize problem state
        self.problem_state = ProblemState(scenario=self.scenario)
        self.entity_count = 0
        
        # Generate first entity
        self.current_entity = self.entity_generator.generate_entity(self.scenario)
        self.entity_count += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute action and return next state."""
        if self.problem_state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")
        
        # Store previous state for reward calculation
        prev_state = ProblemState(
            scenario=self.scenario,
            accepted_count=self.problem_state.accepted_count,
            rejected_count=self.problem_state.rejected_count,
            attribute_counts=self.problem_state.attribute_counts.copy()
        )
        
        # Execute action
        if action == 1:  # ACCEPT
            self.problem_state.accepted_count += 1
            for attr, value in self.current_entity.attributes.items():
                if value:
                    self.problem_state.attribute_counts[attr] = (
                        self.problem_state.attribute_counts.get(attr, 0) + 1
                    )
        else:  # REJECT
            self.problem_state.rejected_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(action, self.current_entity, prev_state, self.problem_state)
        
        # Check if episode is done
        done = self.problem_state.is_terminal() or self.entity_count >= self.max_entities
        
        # Generate next entity if not done
        if not done:
            self.current_entity = self.entity_generator.generate_entity(self.scenario)
            self.entity_count += 1
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info  # done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with FIXED positional encoding for ALL attributes.
        
        This creates a consistent state space where each attribute always has the same position,
        regardless of whether it's constrained in the current scenario.
        """
        state_components = []
        
        # Basic state information (always first 4 components)
        capacity_util = self.problem_state.accepted_count / self.scenario.capacity
        reject_rate = self.problem_state.rejected_count / max(self.scenario.max_rejections, 1)
        remaining_capacity = (self.scenario.capacity - self.problem_state.accepted_count) / self.scenario.capacity
        
        total_processed = self.problem_state.accepted_count + self.problem_state.rejected_count
        progress_ratio = min(total_processed / (self.scenario.capacity + self.scenario.max_rejections), 1.0)
        
        state_components.extend([capacity_util, reject_rate, remaining_capacity, progress_ratio])
        
        # Create constraint lookup for current scenario
        scenario_constraints = {constraint.attribute: constraint for constraint in self.scenario.constraints}
        
        # FIXED POSITIONAL ENCODING: For each attribute in fixed order
        remaining_entities_estimate = max(remaining_capacity * self.config.remaining_entities_multiplier, 
                                        self.config.min_remaining_entities)
        
        for attr in self.fixed_attributes:
            # 1. Current percentage for this attribute (normalized)
            current_pct = self.problem_state.get_attribute_percentage(attr)
            if attr in scenario_constraints:
                constraint = scenario_constraints[attr]
                normalized_pct = current_pct / constraint.min_percentage if constraint.min_percentage > 0 else current_pct
                state_components.append(min(normalized_pct, self.config.max_normalized_value))
            else:
                # Not constrained in this scenario - use raw percentage
                state_components.append(current_pct)
            
            # 2. Deficit for this attribute
            if attr in scenario_constraints:
                constraint = scenario_constraints[attr]
                deficit = max(0, constraint.min_percentage - current_pct)
                normalized_deficit = deficit / constraint.min_percentage if constraint.min_percentage > 0 else deficit
                state_components.append(normalized_deficit)
            else:
                # Not constrained - no deficit
                state_components.append(0.0)
            
            # 3. Urgency score for this attribute
            if attr in scenario_constraints and remaining_capacity > 0:
                constraint = scenario_constraints[attr]
                deficit = max(0, constraint.min_percentage - current_pct)
                
                if deficit > 0:
                    entities_needed = deficit * self.scenario.capacity
                    attr_prob = self.scenario.attribute_probabilities.get(attr, 0.3)
                    expected_helpful = remaining_entities_estimate * attr_prob
                    urgency = entities_needed / max(expected_helpful, 1)
                else:
                    urgency = 0.0
                    
                state_components.append(min(urgency, self.config.max_urgency_score))
            else:
                # Not constrained or no capacity left - no urgency
                state_components.append(0.0)
            
            # 4. Satisfaction indicator for this attribute
            if attr in scenario_constraints:
                constraint = scenario_constraints[attr]
                current_count = self.problem_state.attribute_counts.get(attr, 0)
                is_satisfied = constraint.is_satisfied(self.problem_state.accepted_count, current_count)
                state_components.append(1.0 if is_satisfied else 0.0)
            else:
                # Not constrained - consider satisfied
                state_components.append(1.0)
        
        # Current entity attributes (binary encoding) - FIXED POSITIONAL ORDER
        for attr in self.fixed_attributes:
            has_attr = self.current_entity.attributes.get(attr, False) if self.current_entity else False
            state_components.append(1.0 if has_attr else 0.0)
        
        # Temporal information (always last 2 components)
        entities_processed_ratio = total_processed / self.max_entities if self.max_entities > 0 else 0.0
        estimated_remaining = max(0, self.max_entities - total_processed)
        estimated_remaining_ratio = estimated_remaining / self.max_entities if self.max_entities > 0 else 0.0
        
        state_components.extend([entities_processed_ratio, estimated_remaining_ratio])
        
        return np.array(state_components, dtype=np.float32)
    
    def _get_valid_action_mask(self) -> np.ndarray:
        """Return mask for valid actions (prevent invalid actions like accepting when full)."""
        if self.problem_state.accepted_count >= self.scenario.capacity:
            return np.array([1, 0], dtype=np.float32)  # Only REJECT valid
        return np.array([1, 1], dtype=np.float32)  # Both actions valid
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary with enhanced debugging information."""
        # Calculate which constraints are violated
        violated_constraints = []
        for constraint in self.scenario.constraints:
            current_pct = self.problem_state.get_attribute_percentage(constraint.attribute)
            if current_pct < constraint.min_percentage:
                violated_constraints.append({
                    'attribute': constraint.attribute,
                    'current': current_pct,
                    'required': constraint.min_percentage,
                    'deficit': constraint.min_percentage - current_pct
                })
        
        return {
            'accepted_count': self.problem_state.accepted_count,
            'rejected_count': self.problem_state.rejected_count,
            'constraints_satisfied': self.problem_state.is_satisfied(),
            'capacity_utilization': self.problem_state.accepted_count / self.scenario.capacity,
            'constraint_percentages': {
                constraint.attribute: self.problem_state.get_attribute_percentage(constraint.attribute)
                for constraint in self.scenario.constraints
            },
            'violated_constraints': violated_constraints,
            'num_violated_constraints': len(violated_constraints),
            'success': self.problem_state.is_success(),
            'entity_count': self.entity_count,
            'valid_actions': self._get_valid_action_mask().tolist()
        }
    
    def _calculate_reward(self, action: int, entity: Entity, 
                         prev_state: ProblemState, next_state: ProblemState) -> float:
        """Calculate reward for the given action with improved constraint guidance."""
        reward = 0.0
        
        # Calculate urgency and progress metrics
        remaining_capacity = self.scenario.capacity - prev_state.accepted_count
        capacity_utilization = prev_state.accepted_count / self.scenario.capacity
        
        if action == 1:  # ACCEPT
            # Base capacity reward (scaled by remaining capacity to encourage efficiency)
            reward += self.config.capacity_reward_weight * (1.0 + capacity_utilization)
            
            # Enhanced constraint-specific rewards with urgency weighting
            constraint_improvements = 0.0
            constraint_violations = 0.0
            
            for constraint in self.scenario.constraints:
                prev_pct = prev_state.get_attribute_percentage(constraint.attribute)
                next_pct = next_state.get_attribute_percentage(constraint.attribute)
                
                # Calculate how much this constraint needs to be satisfied
                deficit = max(0, constraint.min_percentage - prev_pct)
                urgency = deficit / max(remaining_capacity / self.scenario.capacity, 0.01)  # Higher when less capacity left
                
                if entity.attributes.get(constraint.attribute, False):
                    if deficit > 0:
                        # Strong reward for helping unsatisfied constraints, scaled by urgency
                        improvement = min(next_pct - prev_pct, deficit)
                        constraint_improvements += improvement * (1.0 + urgency * self.config.urgency_scaling_factor)
                    else:
                        # Small reward for maintaining satisfied constraints
                        constraint_improvements += 0.1
                else:
                    # Penalty for diluting constraint percentages when they're critical
                    if deficit > 0 and urgency > self.config.dilution_urgency_threshold:
                        constraint_violations += urgency * self.config.critical_urgency_scaling
                    elif prev_pct < constraint.min_percentage * self.config.constraint_satisfaction_threshold_multiplier:
                        constraint_violations += self.config.dilution_maintenance_penalty
            
            reward += self.config.constraint_reward_weight * constraint_improvements
            reward -= self.config.dilution_penalty_weight * constraint_violations
            
            # Progress reward: bonus for making progress toward constraint satisfaction
            progress_bonus = 0.0
            for constraint in self.scenario.constraints:
                prev_satisfied = constraint.is_satisfied(prev_state.accepted_count, 
                                                        prev_state.attribute_counts.get(constraint.attribute, 0))
                next_satisfied = constraint.is_satisfied(next_state.accepted_count,
                                                        next_state.attribute_counts.get(constraint.attribute, 0))
                if not prev_satisfied and next_satisfied:
                    progress_bonus += self.config.progress_bonus_scale  # Significant bonus for satisfying a constraint
            
            reward += progress_bonus
        
        else:  # REJECT
            # Dynamic rejection penalty based on context
            base_rejection_penalty = self.config.base_rejection_penalty
            
            # Higher penalty if we're rejecting entities that could help critical constraints
            critical_penalty = 0.0
            for constraint in self.scenario.constraints:
                prev_pct = prev_state.get_attribute_percentage(constraint.attribute)
                deficit = max(0, constraint.min_percentage - prev_pct)
                
                if entity.attributes.get(constraint.attribute, False) and deficit > 0:
                    # Penalty for rejecting helpful entities, scaled by deficit and remaining capacity
                    urgency = deficit / max(remaining_capacity / self.scenario.capacity, 0.01)
                    critical_penalty += urgency * self.config.critical_urgency_scaling
            
            # Penalty scales with remaining capacity (less penalty when capacity is low)
            capacity_penalty = base_rejection_penalty * (remaining_capacity / self.scenario.capacity)
            
            reward -= (capacity_penalty + critical_penalty)
        
        # Enhanced terminal rewards
        if next_state.is_terminal():
            if next_state.is_success():
                # Success bonus scaled by efficiency (how full we got)
                efficiency_bonus = next_state.accepted_count / self.scenario.capacity
                reward += self.config.terminal_reward_scale * (1.0 + efficiency_bonus)
            else:
                # More nuanced failure penalties
                failure_penalty = 0.0
                
                for constraint in self.scenario.constraints:
                    final_pct = next_state.get_attribute_percentage(constraint.attribute)
                    if final_pct < constraint.min_percentage:
                        # Penalty proportional to how far we missed the constraint
                        violation_severity = (constraint.min_percentage - final_pct)
                        failure_penalty += violation_severity * 2.0
                
                # Additional penalty for low capacity utilization
                if next_state.accepted_count < self.scenario.capacity * self.config.low_capacity_threshold:
                    failure_penalty += 1.0
                
                reward -= self.config.terminal_reward_scale * failure_penalty
        
        # Clip reward to reasonable range for training stability
        reward = np.clip(reward, self.config.reward_clip_min, self.config.reward_clip_max)
        return reward
    
    def render(self, mode='human'):
        """Render environment state (optional)."""
        if self.problem_state is None:
            return
        
        print(f"Capacity: {self.problem_state.accepted_count}/{self.scenario.capacity}")
        print(f"Rejected: {self.problem_state.rejected_count}/{self.scenario.max_rejections}")
        print("Constraints:")
        for constraint in self.scenario.constraints:
            current_pct = self.problem_state.get_attribute_percentage(constraint.attribute)
            status = "✓" if current_pct >= constraint.min_percentage else "✗"
            print(f"  {constraint.attribute}: {current_pct:.1%} >= {constraint.min_percentage:.1%} {status}")


class ConstraintAwareCallback(BaseCallback):
    """Enhanced callback for monitoring constraint satisfaction and other metrics during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.constraint_satisfaction_history = []
        self.success_rate_history = []
        self.capacity_utilization_history = []
        self.constraint_violation_history = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each episode with enhanced metrics tracking."""
        if 'episode' in self.locals:
            info = self.locals.get('infos', [{}])[0]
            if info:
                self.constraint_satisfaction_history.append(info.get('constraints_satisfied', False))
                self.success_rate_history.append(info.get('success', False))
                self.capacity_utilization_history.append(info.get('capacity_utilization', 0.0))
                self.constraint_violation_history.append(info.get('num_violated_constraints', 0))
                
                # Log every 100 episodes with enhanced metrics
                if len(self.constraint_satisfaction_history) % 100 == 0:
                    recent_constraint_rate = np.mean(self.constraint_satisfaction_history[-100:])
                    recent_success_rate = np.mean(self.success_rate_history[-100:])
                    recent_capacity_util = np.mean(self.capacity_utilization_history[-100:])
                    recent_violations = np.mean(self.constraint_violation_history[-100:])
                    
                    if self.verbose > 0:
                        print(f"Episode {len(self.constraint_satisfaction_history)}: "
                              f"Constraint: {recent_constraint_rate:.2%}, "
                              f"Success: {recent_success_rate:.2%}, "
                              f"Capacity: {recent_capacity_util:.2%}, "
                              f"Violations: {recent_violations:.1f}")


class DQNStrategy(AbstractStrategy):
    """Stable-Baselines3 DQN strategy for stochastic constraint satisfaction."""
    
    def __init__(self, scenario: Scenario, config: DQNConfig):
        super().__init__(name=f"SB3-DQN-{config.total_timesteps//1000}k")
        self.scenario = scenario
        self.config = config
        
        self.env = SCSPEnvironment(scenario, config)
        
        # Validate environment
        try:
            check_env(self.env)
        except Exception as e:
            print(f"Environment validation warning: {e}")
        
        # Create SB3 DQN model
        self.model = None
        self._create_model()
        
        # Load pre-trained model if provided
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.load_model(self.config.model_path)
        
        # Training history
        self.training_history = {
            'constraint_satisfaction_rates': [],
            'success_rates': [],
            'episode_rewards': []
        }
        
    def _create_model(self):
        """Create SB3 DQN model with configuration."""
        
        import torch.nn as nn
        import torch
        
        # Prepare policy kwargs with enhanced stability features
        policy_kwargs = {
            'net_arch': self.config.net_arch,
            'activation_fn': nn.ReLU,
            'normalize_images': False,  # We handle normalization in observations
            'optimizer_class': torch.optim.Adam,
            'optimizer_kwargs': {
                'eps': 1e-5,  # For numerical stability
                'weight_decay': 1e-4  # L2 regularization
            }
        }
        
        # Create model
        self.model = DQN(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            target_update_interval=self.config.target_update_interval,
            exploration_fraction=self.config.exploration_fraction,
            exploration_initial_eps=self.config.exploration_initial_eps,
            exploration_final_eps=self.config.exploration_final_eps,
            optimize_memory_usage=self.config.optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            verbose=self.config.verbose,
            seed=self.config.seed
        )
    
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using trained SB3 DQN model (optimized version)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_agent() first.")
        
        # Check if we should use the base strategy's optimization
        # This ensures we benefit from automatic acceptance once constraints are satisfied
        if self.should_accept_all_remaining(problem_state, constraints):
            return Decision.ACCEPT
        
        # Store current environment state temporarily (more efficient than creating new env)
        old_state = self.env.problem_state
        old_entity = self.env.current_entity
        old_entity_count = self.env.entity_count
        
        try:
            # Set state for prediction
            self.env.problem_state = problem_state
            self.env.current_entity = entity
            self.env.entity_count = problem_state.accepted_count + problem_state.rejected_count
            
            # Get observation and valid action mask
            observation = self.env._get_observation()
            action_mask = self.env._get_valid_action_mask()
            
            # Predict action with deterministic policy
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Apply action mask to ensure valid action
            if action_mask[action] == 0:
                # If predicted action is invalid, choose the valid one
                valid_actions = np.where(action_mask == 1)[0]
                action = valid_actions[0] if len(valid_actions) > 0 else 0  # Default to REJECT
            
            return Decision.ACCEPT if action == 1 else Decision.REJECT
        
        finally:
            # Always restore original state
            self.env.problem_state = old_state
            self.env.current_entity = old_entity
            self.env.entity_count = old_entity_count
    
    def _reset_internal_state(self) -> None:
        """Reset DQN strategy internal state."""
        # DQN doesn't maintain episode-specific internal state
        # The model state is preserved across episodes for continued learning
        pass
    
    def train_agent(self, total_timesteps: Optional[int] = None) -> Dict[str, List[float]]:
        """Train the SB3 DQN agent."""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
        
        # Create callback for monitoring
        callback = ConstraintAwareCallback(verbose=self.config.verbose)
        
        # Create evaluation environment if specified
        eval_callback = None
        if self.config.eval_freq > 0:
            eval_env = SCSPEnvironment(self.scenario, self.config)
            eval_callback = EvalCallback(
                eval_env, 
                best_model_save_path=None,
                log_path=None,
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True,
                render=False
            )
        
        # Train the model
        if eval_callback:
            self.model.learn(
                total_timesteps=total_timesteps, 
                callback=[callback, eval_callback],
                progress_bar=True
            )
        else:
            self.model.learn(
                total_timesteps=total_timesteps, 
                callback=callback,
                progress_bar=True
            )
        
        # Extract training history
        self.training_history['constraint_satisfaction_rates'] = callback.constraint_satisfaction_history
        self.training_history['success_rates'] = callback.success_rate_history
        
        return self.training_history
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load a pre-trained model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model and set environment
        self.model = DQN.load(path, env=self.env)
    
    def evaluate_agent(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate the trained agent."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_agent() first.")
        
        success_count = 0
        constraint_satisfaction_count = 0
        total_rewards = []
        capacity_utilizations = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                done = done or truncated
            
            total_rewards.append(episode_reward)
            capacity_utilizations.append(info['capacity_utilization'])
            
            if info['success']:
                success_count += 1
            if info['constraints_satisfied']:
                constraint_satisfaction_count += 1
        
        return {
            'success_rate': success_count / num_episodes,
            'constraint_satisfaction_rate': constraint_satisfaction_count / num_episodes,
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_capacity_utilization': np.mean(capacity_utilizations),
            'std_capacity_utilization': np.std(capacity_utilizations)
        }


# Factory function for easy instantiation
def create_dqn_strategy(scenario: Scenario, 
                       learning_rate: float = 0.0005,
                       total_timesteps: int = 100000,
                       model_path: Optional[str] = None) -> DQNStrategy:
    """Create a SB3 DQN strategy with reasonable defaults."""
    config = DQNConfig(
        learning_rate=learning_rate,
        total_timesteps=total_timesteps,
        model_path=model_path
    )
    return DQNStrategy(scenario, config)