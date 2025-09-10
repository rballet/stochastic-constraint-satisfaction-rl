"""
ICU admission environment for reinforcement learning.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ...core.types import ProblemState, Entity, Scenario
from ...simulation.engine import SimulationEngine
from .patient_generator import create_icu_patient_generator
from .config import EnvironmentConfig, RewardConfig, create_config


class ICUProblemState(ProblemState):
    """ICU-specific problem state with medical capacity constraints."""
    
    def __init__(self, env_config: EnvironmentConfig):
        super().__init__()
        self.capacity = env_config.capacity
        self.max_rejections = env_config.max_rejections


class ICUAdmissionEnv:
    """
    ICU Admission Environment for RL training.
    
    This environment wraps the generic simulation framework
    to provide a standardized interface for ICU bed allocation.
    """
    
    def __init__(self, scenario: Scenario, env_config: EnvironmentConfig = None, reward_config: RewardConfig = None, seed: Optional[int] = None):
        """
        Initialize ICU admission environment.
        
        Args:
            scenario: ICU scenario
            env_config: Environment configuration
            reward_config: Reward configuration  
            seed: Random seed
        """
        self.scenario = scenario
        
        if env_config is None or reward_config is None:
            default_env, default_reward = create_config()
            self.env_config = env_config or default_env
            self.reward_config = reward_config or default_reward
        else:
            self.env_config = env_config
            self.reward_config = reward_config
            
        self.seed = seed
        self.patient_generator = create_icu_patient_generator("multivariate", seed)
        self.problem_state = None
        self.current_patient = None
        self.patient_count = 0
        self.observation_space_size = self._calculate_observation_size()
        self.action_space_size = self.env_config.action_space_size
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode."""
        self.problem_state = ICUProblemState(self.env_config)
        self.patient_generator.reset(self.seed)
        self.patient_count = 0
        self.current_patient = self._generate_next_patient()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 0 for reject, 1 for accept
            
        Returns:
            observation, reward, done, info
        """
        # Apply action
        if action == 1:  # Accept
            self._accept_patient()
        else:  # Reject
            self._reject_patient()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.problem_state.is_terminal()
        
        # Generate next patient if not done
        if not done:
            self.current_patient = self._generate_next_patient()
        
        # Prepare info
        info = {
            "accepted_count": self.problem_state.accepted_count,
            "rejected_count": self.problem_state.rejected_count,
            "constraints_satisfied": self._check_constraints_satisfied(),
            "patient_count": self.patient_count,
        }
        
        return self._get_observation(), reward, done, info
    
    def _generate_next_patient(self) -> Entity:
        """Generate the next patient."""
        self.patient_count += 1
        patient = self.patient_generator.generate_entity(self.scenario)
        patient.id = self.patient_count
        return patient
    
    def _accept_patient(self):
        """Accept the current patient."""
        self.problem_state.accepted_count += 1
        
        # Update attribute counts
        for attr, value in self.current_patient.attributes.items():
            if value:
                self.problem_state.attribute_counts[attr] = (
                    self.problem_state.attribute_counts.get(attr, 0) + 1
                )
    
    def _reject_patient(self):
        """Reject the current patient."""
        self.problem_state.rejected_count += 1
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Observation includes:
        - Current patient attributes (binary)
        - ICU state (beds used, rejections)
        - Constraint satisfaction status
        - Remaining capacity
        """
        if self.current_patient is None:
            patient_attrs = [0] * len(self.scenario.attributes)
        else:
            patient_attrs = [
                1 if self.current_patient.attributes.get(attr, False) else 0
                for attr in self.scenario.attributes
            ]
        
        # ICU state
        icu_state = [
            self.problem_state.accepted_count / self.env_config.capacity,  # Normalized bed occupancy
            self.problem_state.rejected_count / self.env_config.max_rejections,  # Normalized rejections
            (self.env_config.capacity - self.problem_state.accepted_count) / self.env_config.capacity,  # Remaining capacity
        ]
        
        # Constraint satisfaction status
        constraint_status = []
        for constraint in self.scenario.constraints:
            if self.problem_state.accepted_count > 0:
                current_pct = self.problem_state.get_attribute_percentage(constraint.attribute)
                constraint_status.extend([
                    current_pct,  # Current percentage
                    max(0, constraint.min_percentage - current_pct),  # Deficit
                ])
            else:
                constraint_status.extend([0.0, constraint.min_percentage])
        
        # Current attribute percentages
        attr_percentages = []
        for attr in self.scenario.attributes:
            if self.problem_state.accepted_count > 0:
                pct = self.problem_state.get_attribute_percentage(attr)
            else:
                pct = 0.0
            attr_percentages.append(pct)
        
        observation = np.array(
            patient_attrs + icu_state + constraint_status + attr_percentages,
            dtype=np.float32
        )
        
        return observation
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward for the current step.
        
        Reward structure:
        - Positive reward for accepting patients that help meet constraints
        - Negative reward for rejecting patients when there's capacity
        - Bonus for successfully filling ICU while meeting constraints
        - Penalty for constraint violations
        """
        reward = 0.0
        
        # Base reward for making progress toward capacity
        if self.problem_state.accepted_count > 0:
            capacity_reward = self.problem_state.accepted_count / self.env_config.capacity
            reward += capacity_reward * self.reward_config.capacity_weight
        
        # Additional reward components
        if self.reward_config.acceptance_reward > 0:
            if self.problem_state.accepted_count > 0:  # Patient was just accepted
                reward += self.reward_config.acceptance_reward
        
        # Constraint satisfaction rewards/penalties
        constraint_penalty = 0.0
        for constraint in self.scenario.constraints:
            if self.problem_state.accepted_count > 0:
                current_pct = self.problem_state.get_attribute_percentage(constraint.attribute)
                if current_pct < constraint.min_percentage:
                    deficit = constraint.min_percentage - current_pct
                    constraint_penalty += deficit * self.reward_config.constraint_penalty
        
        reward -= constraint_penalty
        
        # Early constraint satisfaction bonus
        if (self.reward_config.early_bonus > 0 and
            self._check_constraints_satisfied() and 
            self.problem_state.accepted_count < self.env_config.capacity):
            reward += self.reward_config.early_bonus
        
        # Success bonus
        if self.problem_state.is_full() and self._check_constraints_satisfied():
            reward += self.reward_config.success_bonus
        
        # Failure penalty
        if self.problem_state.max_rejections_reached():
            reward -= self.reward_config.failure_penalty
        
        # Apply dynamic scaling
        if self.reward_config.use_dynamic_scaling:
            reward *= self.reward_config.scaling_factor
        
        return reward
    
    def _check_constraints_satisfied(self) -> bool:
        """Check if all constraints are currently satisfied."""
        if self.problem_state.accepted_count == 0:
            return False
        
        for constraint in self.scenario.constraints:
            current_count = self.problem_state.attribute_counts.get(constraint.attribute, 0)
            if not constraint.is_satisfied(self.problem_state.accepted_count, current_count):
                return False
        
        return True
    
    def _calculate_observation_size(self) -> int:
        """Calculate the size of the observation space."""
        # Patient attributes + ICU state + constraint status + attribute percentages
        patient_attrs = len(self.scenario.attributes)
        icu_state = 3  # bed occupancy, rejections, remaining capacity
        constraint_status = len(self.scenario.constraints) * 2  # current + deficit
        attr_percentages = len(self.scenario.attributes)
        
        return patient_attrs + icu_state + constraint_status + attr_percentages
    
    def render(self, mode='human'):
        """Render the current state."""
        print(f"ICU Status: {self.problem_state.accepted_count}/{self.env_config.capacity} beds occupied")
        print(f"Rejections: {self.problem_state.rejected_count}/{self.env_config.max_rejections}")
        
        if self.current_patient:
            print(f"Current Patient #{self.current_patient.id}:")
            for attr, value in self.current_patient.attributes.items():
                print(f"  {attr}: {'Yes' if value else 'No'}")
        
        print("Constraint Status:")
        for constraint in self.scenario.constraints:
            if self.problem_state.accepted_count > 0:
                current_pct = self.problem_state.get_attribute_percentage(constraint.attribute)
                status = "✓" if current_pct >= constraint.min_percentage else "✗"
                print(f"  {constraint.description}: {current_pct:.2%} (req: {constraint.min_percentage:.2%}) {status}")
            else:
                print(f"  {constraint.description}: 0% (req: {constraint.min_percentage:.2%}) ✗")
        print()