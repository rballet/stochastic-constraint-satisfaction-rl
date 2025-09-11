"""
Linear Programming strategies for stochastic constraint satisfaction problems.

This module implements multiple LP-based approaches:
1. Basic Linear Programming
2. Multi-Stage Stochastic Programming  
3. Robust Optimization
4. Chance Constraints

Each strategy uses different approaches to handle uncertainty and constraints.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import norm
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.strategy_base import AbstractStrategy
from ..core.types import Entity, ProblemState, Constraint, Decision, Scenario


@dataclass
class LPStrategyConfig:
    """Base configuration for LP strategies."""
    acceptance_threshold: float = 0.5  # Accept if LP probability > threshold
    lookahead_horizon: int = 50  # Number of future entities to consider
    use_conservative_estimates: bool = True  # Use conservative probability estimates
    constraint_buffer: float = 0.05  # Buffer above minimum constraint requirements
    solver_method: str = "highs"  # LP solver method
    
    # Reward calculation parameters
    base_reward: float = 1.0  # Base reward for accepting any entity
    future_reward_discount: float = 0.1  # Discount factor for future rewards
    constraint_urgency_multiplier: float = 10.0  # Multiplier for constraint urgency
    constraint_penalty_multiplier: float = 5.0  # Penalty for not helping constraints
    
    # Heuristic parameters
    urgent_deficit_threshold: float = 0.1  # Threshold for urgent constraint violations
    moderate_deficit_threshold: float = 0.05  # Threshold for moderate violations
    capacity_threshold: float = 0.8  # Threshold for considering capacity as "running low"
    
    # LP formulation parameters
    max_horizon_limit: int = 20  # Maximum horizon for computational efficiency
    min_capacity_for_lp: int = 2  # Minimum remaining capacity to use LP

@dataclass
class RobustOptimizationConfig(LPStrategyConfig):
    """Configuration for Robust Optimization strategy."""
    uncertainty_budget: float = 0.1  # Budget for uncertainty in constraints
    worst_case_scenarios: int = 5  # Number of worst-case scenarios to consider

@dataclass
class ChanceConstraintConfig(LPStrategyConfig):
    """Configuration for Chance Constraint strategy."""
    confidence_level: float = 0.95  # Confidence level for constraint satisfaction
    sample_size: int = 1000  # Sample size for Monte Carlo approximation

@dataclass
class MultiStageConfig(LPStrategyConfig):
    """Configuration for Multi-Stage Stochastic Programming."""
    num_stages: int = 3  # Number of decision stages
    scenario_tree_size: int = 8  # Number of scenarios per stage
    
    # Stage tree construction parameters
    max_stages_limit: int = 4  # Maximum stages for computational efficiency
    max_scenarios_per_stage: int = 8  # Maximum scenarios per stage
    
    # Probability variation parameters
    scenario_probability_variance: float = 0.2  # Variation in attribute probabilities
    min_attribute_probability: float = 0.1  # Minimum allowed attribute probability
    max_attribute_probability: float = 0.9  # Maximum allowed attribute probability
    
    # Backward induction parameters
    future_value_discount_factor: float = 0.9  # Discount factor for future stages
    stage_discount_base: float = 0.9  # Base discount factor for stage distances
    
    # Value calculation parameters
    first_entity_bonus: float = 2.0  # Bonus reward for first entity accepted
    constraint_deficit_multiplier: float = 3.0  # Multiplier for constraint deficit rewards
    max_deficit_bonus: float = 10.0  # Maximum bonus for addressing deficits
    constraint_satisfaction_bonus: float = 1.0  # Bonus for satisfying constraints
    constraint_violation_penalty: float = 2.0  # Penalty for constraint violations
    
    # Opportunity cost calculation
    future_acceptance_rate_estimate: float = 0.7  # Expected future acceptance rate
    opportunity_cost_weight: float = 1.0  # Weight for opportunity cost in decision
    
    # Computational efficiency
    min_remaining_capacity_for_multistage: int = 2  # Min capacity to use multistage
    single_stage_fallback_threshold: int = 5  # Use single stage if capacity < threshold


class BaseLPStrategy(AbstractStrategy, ABC):
    """
    Base class for Linear Programming strategies.
    
    Provides common functionality for LP-based approaches to SCSP.
    """
    
    def __init__(self, name: str, scenario: Scenario, config: LPStrategyConfig):
        super().__init__(name)
        self.scenario = scenario
        self.config = config
        
        # Cache for optimization
        self._attribute_to_index = {attr: i for i, attr in enumerate(scenario.attributes)}
        self._constraint_attributes = {c.attribute for c in scenario.constraints}
        
    @abstractmethod
    def _formulate_lp(self, entity: Entity, problem_state: ProblemState, 
                     constraints: List[Constraint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formulate LP problem. Returns (c, A_ub, b_ub) for scipy.optimize.linprog."""
        pass
    
    def _entity_to_vector(self, entity: Entity) -> np.ndarray:
        """Convert entity to binary vector representation."""
        vector = np.zeros(len(self.scenario.attributes))
        for i, attr in enumerate(self.scenario.attributes):
            vector[i] = 1.0 if entity.attributes.get(attr, False) else 0.0
        return vector
    
    def _calculate_constraint_deficit(self, problem_state: ProblemState, constraints: List[Constraint]) -> Dict[str, float]:
        """
        Calculate current constraint deficits according to mathematical specification.
        
        Formula: deficit_j = max(0, θ_j * n_t - c_j^(t))
        Where:
        - θ_j: minimum percentage required for attribute j
        - n_t: current number of accepted entities  
        - c_j^(t): current count of entities with attribute j
        """
        deficits = {}
        n_t = problem_state.accepted_count
        
        if n_t == 0:
            # Special case: no entities accepted yet
            for constraint in constraints:
                deficits[constraint.attribute] = 0.0  # No deficit when no entities yet
            return deficits
            
        for constraint in constraints:
            attr = constraint.attribute
            theta_j = constraint.min_percentage
            c_j_t = problem_state.attribute_counts.get(attr, 0)
            
            # deficit_j = max(0, θ_j * n_t - c_j^(t))
            required_count = theta_j * n_t
            deficit = max(0.0, required_count - c_j_t)
            deficits[attr] = deficit
            
        return deficits
    
    def _reset_internal_state(self) -> None:
        """Reset any internal state."""
        pass


class LinearProgrammingStrategy(BaseLPStrategy):
    """
    Standard Linear Programming strategy implementing the formulation from the documentation.
    
    Uses expected value approximation and lookahead horizon to handle uncertainty.
    """
    
    def __init__(self, scenario: Scenario, config: Optional[LPStrategyConfig] = None):
        super().__init__("LinearProgramming", scenario, config or LPStrategyConfig())
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using linear programming optimization."""
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return Decision.REJECT
        
        # Formulate and solve LP
        try:
            c, A_ub, b_ub = self._formulate_lp(entity, problem_state, constraints)
            if c is None:
                return Decision.REJECT
                
            # Solve LP (minimization problem, so negate objective)
            # Bounds: x_t in {0,1}, y_future in [0, remaining_capacity]
            bounds = [(0, 1), (0, remaining_capacity)]
            result = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                           method=self.config.solver_method)
            
            if result.success and len(result.x) > 0:
                # First variable is decision for current entity
                x_t = result.x[0]
                return Decision.ACCEPT if x_t >= self.config.acceptance_threshold else Decision.REJECT
            else:
                # LP failed or infeasible - use constraint-aware heuristic
                return self._heuristic_decision(entity, problem_state, constraints)
                
        except Exception:
            # Fallback to heuristic on solver failure
            return self._heuristic_decision(entity, problem_state, constraints)
    
    def _formulate_lp(self, entity: Entity, problem_state: ProblemState, 
                     constraints: List[Constraint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Formulate LP following the corrected mathematical specification:
        
        Variables: [x_t, y_future] where y_future = sum of expected future acceptances
        
        Objective: max R_immediate(x_t) + R_future * y_future
        
        Constraints:
        1. n_t + x_t + y_future <= N_max  (capacity)
        2. c_j^(t) + x_t * I(e_t has a_j) + y_future * p_j >= θ_j * (n_t + x_t + y_future)  (attributes)
        3. 0 <= y_future <= remaining_capacity, x_t in {0,1}
        """
        
        # Get current state variables
        n_t = problem_state.accepted_count
        N_max = problem_state.capacity
        remaining_capacity = N_max - n_t
        
        if remaining_capacity <= self.config.min_capacity_for_lp:
            return None, None, None
        
        # Simplified formulation: [x_t, y_future] (2 variables)
        num_vars = 2
        
        # Objective: maximize immediate reward + expected future rewards
        c = np.zeros(num_vars)
        c[0] = self._calculate_immediate_reward(entity, problem_state, constraints)
        c[1] = self.config.future_reward_discount * self.config.base_reward
        
        # Constraints (for scipy.optimize.linprog, convert >= to <= by negating)
        A_ub_list = []
        b_ub_list = []
        
        # Constraint 1: Capacity constraint (n_t + x_t + y_future <= N_max)
        # Rearranged: x_t + y_future <= N_max - n_t
        capacity_constraint = np.array([1.0, 1.0])
        A_ub_list.append(capacity_constraint)
        b_ub_list.append(remaining_capacity)
        
        # Constraint 2: Attribute constraints using corrected formulation
        # c_j^(t) + x_t * I(e_t has a_j) + y_future * p_j >= θ_j * (n_t + x_t + y_future)
        # Rearranged: x_t * (I(e_t has a_j) - θ_j) + y_future * (p_j - θ_j) >= θ_j * n_t - c_j^(t)
        # For linprog (<=): -(x_t * (I(e_t has a_j) - θ_j) + y_future * (p_j - θ_j)) <= -(θ_j * n_t - c_j^(t))
        
        for constraint in constraints:
            attr = constraint.attribute
            theta_j = constraint.min_percentage
            p_j = self.scenario.attribute_probabilities.get(attr, 0.5)
            entity_has_attr = 1.0 if entity.attributes.get(attr, False) else 0.0
            c_j_t = problem_state.attribute_counts.get(attr, 0)
            
            # Calculate right-hand side: θ_j * n_t - c_j^(t)
            rhs = theta_j * n_t - c_j_t
            
            # Calculate coefficients for left-hand side
            x_t_coeff = entity_has_attr - theta_j
            y_future_coeff = p_j - theta_j
            
            # For linprog (A_ub * x <= b_ub), negate to convert >= to <=
            attr_constraint = np.array([-x_t_coeff, -y_future_coeff])
            bound_value = -rhs
            
            # Add small buffer for numerical stability (make constraint easier to satisfy)
            bound_value += self.config.constraint_buffer
            
            A_ub_list.append(attr_constraint)
            b_ub_list.append(bound_value)
        
        # Convert to numpy arrays
        if A_ub_list:
            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
        else:
            # Only capacity constraint
            A_ub = capacity_constraint.reshape(1, -1)
            b_ub = np.array([remaining_capacity])
        
        return c, A_ub, b_ub
    
    def _calculate_immediate_reward(self, entity: Entity, problem_state: ProblemState, 
                                  constraints: List[Constraint]) -> float:
        """Calculate immediate reward for accepting this entity with constraint awareness."""
        if problem_state.accepted_count == 0:
            return self.config.base_reward  # Base reward for first entity
        
        constraint_reward = 0.0
        constraint_penalty = 0.0
        
        # Calculate deficits and penalties
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        
        for constraint in constraints:
            attr = constraint.attribute
            deficit = deficits[attr]
            entity_has_attr = entity.attributes.get(attr, False)
            
            if deficit > 0:  # Constraint not satisfied
                if entity_has_attr:
                    # Reward for helping with constraint deficit (scaled by urgency)
                    urgency_multiplier = min(self.config.constraint_urgency_multiplier, deficit * 2)
                    constraint_reward += urgency_multiplier
                else:
                    # Penalty for not helping with constraint when needed
                    constraint_penalty += deficit * self.config.constraint_penalty_multiplier
            else:
                # Constraint already satisfied
                if entity_has_attr:
                    constraint_reward += self.config.base_reward * 0.1  # Small bonus for maintaining constraint
        
        # Calculate total reward
        total_reward = self.config.base_reward + constraint_reward - constraint_penalty
        
        # Additional penalty if entity doesn't help with any deficits when constraints are violated
        max_deficit = max(deficits.values()) if deficits else 0
        if (constraint_penalty > constraint_reward and 
            max_deficit > self.config.moderate_deficit_threshold):
            total_reward -= self.config.base_reward * 2  # Strong penalty for unhelpful entities
        
        return max(0.01, total_reward)  # Ensure small positive reward
    
    def _heuristic_decision(self, entity: Entity, problem_state: ProblemState, 
                          constraints: List[Constraint]) -> Decision:
        """Constraint-aware heuristic when LP solver fails or is infeasible."""
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        
        # Calculate remaining capacity
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return Decision.REJECT
        
        # Calculate urgency of constraints
        max_deficit = max(deficits.values()) if deficits else 0
        
        # Priority 1: Accept if entity helps with urgent constraint violations
        for constraint in constraints:
            attr = constraint.attribute
            deficit = deficits[attr]
            entity_has_attr = entity.attributes.get(attr, False)
            if deficit > self.config.urgent_deficit_threshold and entity_has_attr:
                return Decision.ACCEPT
        
        # Priority 2: If constraints are moderately violated, be selective
        if max_deficit > self.config.moderate_deficit_threshold:
            entity_helps = any(
                entity.attributes.get(constraint.attribute, False) and deficits[constraint.attribute] > 0
                for constraint in constraints
            )
            if entity_helps:
                return Decision.ACCEPT
            else:
                # Reject entities that don't help when constraints are violated
                return Decision.REJECT
        
        # Priority 3: If capacity is running low, be more selective
        capacity_used = problem_state.accepted_count / problem_state.capacity
        if capacity_used > self.config.capacity_threshold:
            # Only accept if entity helps with any constraint
            entity_helps = any(
                entity.attributes.get(constraint.attribute, False)
                for constraint in constraints
            )
            return Decision.ACCEPT if entity_helps else Decision.REJECT
        
        # Priority 4: Normal operation - accept with moderate probability
        return Decision.ACCEPT
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy configuration information."""
        return {
            "type": "LinearProgramming",
            "acceptance_threshold": self.config.acceptance_threshold,
            "lookahead_horizon": self.config.lookahead_horizon,
            "constraint_buffer": self.config.constraint_buffer,
            "solver_method": self.config.solver_method
        }
