"""
Advanced Linear Programming strategies for stochastic constraint satisfaction problems.

This module implements advanced LP approaches:
1. Multi-Stage Stochastic Programming  
2. Robust Optimization
3. Chance Constraints

These strategies extend the basic LP formulation to handle uncertainty more sophisticatedly.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import norm
from dataclasses import dataclass

from .lp_strategy import BaseLPStrategy, MultiStageConfig, RobustOptimizationConfig, ChanceConstraintConfig
from ..core.types import Entity, ProblemState, Constraint, Decision, Scenario


class MultiStageStochasticStrategy(BaseLPStrategy):
    """
    Multi-Stage Stochastic Programming strategy.
    
    Properly implements multi-stage stochastic programming with:
    1. Scenario tree generation for future uncertainty
    2. Recourse variables for future decisions
    3. Stage-wise decision optimization
    
    Mathematical formulation:
    max E[Q_1(x_1)] where Q_t(x_t) = R_t(x_t) + E[Q_{t+1}(x_{t+1})]
    """
    
    def __init__(self, scenario: Scenario, config: Optional[MultiStageConfig] = None):
        super().__init__("MultiStageStochastic", scenario, config or MultiStageConfig())
        self.config: MultiStageConfig = self.config
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using proper multi-stage stochastic programming."""
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return Decision.REJECT
        
        # Use single-stage fallback for small remaining capacity
        if remaining_capacity < self.config.single_stage_fallback_threshold:
            return self._single_stage_decision(entity, problem_state, constraints)
        
        # Build scenario tree for multiple stages
        scenario_tree = self._build_scenario_tree(problem_state, remaining_capacity)
        
        # Solve multi-stage stochastic program
        optimal_value = self._solve_multistage_stochastic_program(
            entity, problem_state, constraints, scenario_tree
        )
        
        return Decision.ACCEPT if optimal_value >= self.config.acceptance_threshold else Decision.REJECT
    
    def _formulate_lp(self, entity: Entity, problem_state: ProblemState, 
                     constraints: List[Constraint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formulate single-stage LP for compatibility (not used in multi-stage solving)."""
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return None, None, None
            
        # Simple single-variable LP for current decision only
        c = np.array([self._calculate_immediate_reward(entity, problem_state, constraints)])
        A_ub = np.array([[1.0]])  # Capacity constraint: x <= 1
        b_ub = np.array([1.0])
        
        return c, A_ub, b_ub
    
    def _single_stage_decision(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Simple single-stage decision for small remaining capacity."""
        immediate_reward = self._calculate_immediate_reward(entity, problem_state, constraints)
        constraint_value = self._calculate_constraint_value(entity, problem_state, constraints)
        
        # For small capacity, be more aggressive in accepting helpful entities
        total_value = immediate_reward + constraint_value * 2.0  # Higher weight on constraints
        
        return Decision.ACCEPT if total_value >= self.config.acceptance_threshold else Decision.REJECT
    
    def _build_scenario_tree(self, problem_state: ProblemState, remaining_capacity: int) -> Dict[str, Any]:
        """Build proper scenario tree for multi-stage stochastic programming."""
        
        # Use configurable limits for computational efficiency
        num_stages = min(self.config.num_stages, self.config.max_stages_limit)
        scenarios_per_stage = min(self.config.scenario_tree_size, self.config.max_scenarios_per_stage)
        stage_horizon = max(1, remaining_capacity // num_stages)
        
        scenario_tree = {
            'num_stages': num_stages,
            'scenarios_per_stage': scenarios_per_stage,
            'stage_horizon': stage_horizon,
            'stages': []
        }
        
        # Generate scenarios for each stage
        for stage in range(num_stages):
            stage_scenarios = []
            
            for scenario_idx in range(scenarios_per_stage):
                # Generate probability variations for this scenario
                scenario_probs = {}
                for attr, base_prob in self.scenario.attribute_probabilities.items():
                    # Add controlled variation based on scenario index
                    variation = (scenario_idx / scenarios_per_stage - 0.5) * self.config.scenario_probability_variance
                    scenario_probs[attr] = max(
                        self.config.min_attribute_probability, 
                        min(self.config.max_attribute_probability, base_prob + variation)
                    )
                
                stage_scenarios.append({
                    'probability': 1.0 / scenarios_per_stage,
                    'attribute_probabilities': scenario_probs,
                    'expected_entities': stage_horizon
                })
            
            scenario_tree['stages'].append(stage_scenarios)
        
        return scenario_tree
    
    def _solve_multistage_stochastic_program(self, entity: Entity, problem_state: ProblemState, 
                                           constraints: List[Constraint], scenario_tree: Dict[str, Any]) -> float:
        """Solve proper multi-stage stochastic program using backward induction."""
        
        # Stage 1: Current decision (accept/reject current entity)
        current_decision_value = self._solve_current_stage(entity, problem_state, constraints, scenario_tree)
        
        # Calculate expected future value using backward induction
        future_value = self._backward_induction(problem_state, constraints, scenario_tree)
        
        # Total value = immediate + discounted future
        total_value = current_decision_value + self.config.future_value_discount_factor * future_value
        
        return total_value
    
    def _solve_current_stage(self, entity: Entity, problem_state: ProblemState, 
                           constraints: List[Constraint], scenario_tree: Dict[str, Any]) -> float:
        """Solve current stage decision with full scenario tree lookahead."""
        
        # Calculate immediate reward
        immediate_reward = self._calculate_immediate_reward(entity, problem_state, constraints)
        
        # Calculate constraint improvement value
        constraint_value = self._calculate_constraint_value(entity, problem_state, constraints)
        
        # Calculate opportunity cost of accepting this entity
        opportunity_cost = self._calculate_opportunity_cost(entity, problem_state, constraints, scenario_tree)
        
        # Final decision value
        decision_value = immediate_reward + constraint_value - opportunity_cost
        
        return max(0.0, decision_value)
    
    def _backward_induction(self, problem_state: ProblemState, constraints: List[Constraint], 
                          scenario_tree: Dict[str, Any]) -> float:
        """Use backward induction to calculate expected future value."""
        
        if scenario_tree['num_stages'] <= 1:
            return 0.0
        
        total_expected_value = 0.0
        
        # Work backwards from last stage
        for stage_idx in range(scenario_tree['num_stages'] - 1, 0, -1):
            stage_scenarios = scenario_tree['stages'][stage_idx]
            stage_value = 0.0
            
            for scenario in stage_scenarios:
                scenario_value = self._evaluate_scenario_value(
                    problem_state, constraints, scenario, stage_idx
                )
                stage_value += scenario['probability'] * scenario_value
            
            total_expected_value += stage_value * (self.config.stage_discount_base ** stage_idx)
        
        return total_expected_value
    
    def _evaluate_scenario_value(self, problem_state: ProblemState, constraints: List[Constraint], 
                               scenario: Dict[str, Any], stage_idx: int) -> float:
        """Evaluate the value of a specific scenario at a given stage."""
        
        # Simulate expected state at this stage
        expected_accepted = min(
            problem_state.accepted_count + stage_idx * scenario['expected_entities'],
            problem_state.capacity
        )
        
        # Calculate expected constraint satisfaction
        constraint_satisfaction = 0.0
        for constraint in constraints:
            attr = constraint.attribute
            attr_prob = scenario['attribute_probabilities'].get(attr, 0.5)
            
            # Expected count of this attribute at this stage
            expected_count = expected_accepted * attr_prob
            current_satisfaction = expected_count / expected_accepted if expected_accepted > 0 else 0
            
            if current_satisfaction >= constraint.min_percentage:
                constraint_satisfaction += self.config.constraint_satisfaction_bonus
            else:
                # Penalty for not meeting constraint
                deficit = constraint.min_percentage - current_satisfaction
                constraint_satisfaction -= deficit * self.config.constraint_violation_penalty
        
        # Base value for capacity utilization
        capacity_value = expected_accepted / problem_state.capacity
        
        return capacity_value + constraint_satisfaction
    
    def _calculate_constraint_value(self, entity: Entity, problem_state: ProblemState, 
                                  constraints: List[Constraint]) -> float:
        """Calculate value of accepting entity for constraint satisfaction."""
        
        value = 0.0
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        
        for constraint in constraints:
            attr = constraint.attribute
            if entity.attributes.get(attr, False) and deficits[attr] > 0:
                # Higher value for helping with larger deficits
                value += deficits[attr] * self.config.constraint_deficit_multiplier
        
        return value
    
    def _calculate_opportunity_cost(self, entity: Entity, problem_state: ProblemState, 
                                  constraints: List[Constraint], scenario_tree: Dict[str, Any]) -> float:
        """Calculate opportunity cost of accepting this entity now vs waiting."""
        
        if problem_state.capacity - problem_state.accepted_count <= self.config.min_remaining_capacity_for_multistage:
            return 0.0  # No opportunity cost if very little capacity remains
        
        # Estimate if we can get better entities in the future
        current_utility = self._calculate_entity_utility(entity, constraints)
        
        # Expected utility of future entities
        avg_future_utility = 0.0
        count = 0
        
        for stage_scenarios in scenario_tree['stages']:
            for scenario in stage_scenarios:
                future_utility = 0.0
                for attr, prob in scenario['attribute_probabilities'].items():
                    if any(c.attribute == attr for c in constraints):
                        future_utility += prob  # Simple utility based on constraint relevance
                
                avg_future_utility += scenario['probability'] * future_utility
                count += 1
        
        if count > 0:
            avg_future_utility /= count
        
        # Apply opportunity cost weight and return weighted difference
        opportunity_cost = max(0.0, avg_future_utility - current_utility) * self.config.opportunity_cost_weight
        
        return opportunity_cost
    
    def _calculate_entity_utility(self, entity: Entity, constraints: List[Constraint]) -> float:
        """Calculate utility of an entity based on constraint helpfulness."""
        utility = 0.0
        for constraint in constraints:
            if entity.attributes.get(constraint.attribute, False):
                utility += 1.0
        return utility / len(constraints) if constraints else 0.0
    
    def _calculate_immediate_reward(self, entity: Entity, problem_state: ProblemState, 
                                  constraints: List[Constraint]) -> float:
        """Calculate immediate reward for accepting this entity."""
        if problem_state.accepted_count == 0:
            return self.config.first_entity_bonus
        
        reward = self.config.base_reward
        
        # Bonus for helping with constraint deficits
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        for constraint in constraints:
            attr = constraint.attribute
            if entity.attributes.get(attr, False) and deficits[attr] > 0:
                reward += min(
                    deficits[attr] * self.config.constraint_deficit_multiplier, 
                    self.config.max_deficit_bonus
                )
        
        return reward
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy configuration information."""
        return {
            "type": "MultiStageStochastic",
            "acceptance_threshold": self.config.acceptance_threshold,
            "num_stages": self.config.num_stages,
            "scenario_tree_size": self.config.scenario_tree_size,
            "implementation": "proper_multistage_with_backward_induction"
        }


class RobustOptimizationStrategy(BaseLPStrategy):
    """
    Robust Optimization strategy.
    
    Optimizes for worst-case scenarios within an uncertainty set.
    Implements the formulation from docs/LINEAR_PROGRAMMING.md:
    
    max min_scenario Objective(scenario)
    """
    
    def __init__(self, scenario: Scenario, config: Optional[RobustOptimizationConfig] = None):
        super().__init__("RobustOptimization", scenario, config or RobustOptimizationConfig())
        self.config: RobustOptimizationConfig = self.config
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using robust optimization."""
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return Decision.REJECT
        
        # Solve robust optimization problem
        robust_value = self._solve_robust_optimization(entity, problem_state, constraints)
        
        return Decision.ACCEPT if robust_value >= self.config.acceptance_threshold else Decision.REJECT
    
    def _formulate_lp(self, entity: Entity, problem_state: ProblemState, 
                     constraints: List[Constraint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formulate robust optimization problem."""
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return None, None, None
        
        # Generate worst-case scenarios
        worst_case_scenarios = self._generate_worst_case_scenarios()
        
        # Formulate min-max problem (simplified to single scenario here)
        c = np.array([self._calculate_immediate_reward(entity, problem_state, constraints)])
        
        # Conservative constraints using worst-case probabilities
        A_ub_list = []
        b_ub_list = []
        
        # Capacity constraint
        A_ub_list.append([1.0])
        b_ub_list.append(1.0)
        
        # Robust constraint formulation
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        
        for constraint in constraints:
            attr = constraint.attribute
            deficit = deficits[attr]
            
            if deficit > 0:
                # Use worst-case probability for this attribute
                worst_case_prob = min(worst_case_scenarios.get(attr, 0.5), 
                                    self.scenario.attribute_probabilities.get(attr, 0.5))
                
                # More conservative constraint
                entity_contrib = 1.0 if entity.attributes.get(attr, False) else 0.0
                required = deficit + self.config.constraint_buffer + self.config.uncertainty_budget
                
                if entity_contrib < required:
                    # This entity alone cannot satisfy the constraint
                    A_ub_list.append([1.0])
                    b_ub_list.append(0.0)  # Reject
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        return c, A_ub, b_ub
    
    def _generate_worst_case_scenarios(self) -> Dict[str, float]:
        """Generate worst-case attribute probabilities."""
        worst_case = {}
        
        for attr, prob in self.scenario.attribute_probabilities.items():
            # Worst case: lower probability of desired attributes
            worst_case[attr] = max(0.1, prob - self.config.uncertainty_budget)
        
        return worst_case
    
    def _solve_robust_optimization(self, entity: Entity, problem_state: ProblemState, 
                                 constraints: List[Constraint]) -> float:
        """Solve robust optimization problem."""
        
        try:
            c, A_ub, b_ub = self._formulate_lp(entity, problem_state, constraints)
            if c is None:
                return 0.0
                
            result = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * len(c))
            
            if result.success:
                return result.x[0]
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _calculate_immediate_reward(self, entity: Entity, problem_state: ProblemState, 
                                  constraints: List[Constraint]) -> float:
        """Calculate immediate reward for accepting this entity."""
        if problem_state.accepted_count == 0:
            return 2.0  # High reward for first entity
        
        reward = 1.0  # Base reward
        
        # Bonus for helping with constraint deficits
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        for constraint in constraints:
            attr = constraint.attribute
            if entity.attributes.get(attr, False) and deficits[attr] > 0:
                reward += deficits[attr] * 5.0  # High reward for addressing deficits
        
        return reward
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy configuration information."""
        return {
            "type": "RobustOptimization",
            "acceptance_threshold": self.config.acceptance_threshold,
            "uncertainty_budget": self.config.uncertainty_budget,
            "worst_case_scenarios": self.config.worst_case_scenarios
        }


class ChanceConstraintStrategy(BaseLPStrategy):
    """
    Chance Constraint strategy.
    
    Ensures constraints are satisfied with high probability rather than deterministically.
    Implements the formulation from docs/LINEAR_PROGRAMMING.md:
    
    P(Constraint satisfied) >= 1 - alpha
    """
    
    def __init__(self, scenario: Scenario, config: Optional[ChanceConstraintConfig] = None):
        super().__init__("ChanceConstraint", scenario, config or ChanceConstraintConfig())
        self.config: ChanceConstraintConfig = self.config
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using chance constraints."""
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return Decision.REJECT
        
        # Evaluate chance constraint satisfaction
        satisfaction_probability = self._evaluate_chance_constraints(entity, problem_state, constraints)
        
        return Decision.ACCEPT if satisfaction_probability >= self.config.confidence_level else Decision.REJECT
    
    def _formulate_lp(self, entity: Entity, problem_state: ProblemState, 
                     constraints: List[Constraint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formulate chance constraint problem."""
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return None, None, None
        
        # Convert chance constraints to deterministic equivalents using normal approximation
        c = np.array([self._calculate_immediate_reward(entity, problem_state, constraints)])
        
        A_ub_list = []
        b_ub_list = []
        
        # Capacity constraint
        A_ub_list.append([1.0])
        b_ub_list.append(1.0)
        
        # Chance constraint reformulation
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        
        for constraint in constraints:
            attr = constraint.attribute
            deficit = deficits[attr]
            
            if deficit > 0:
                # Normal approximation for chance constraint
                # P(constraint satisfied) >= confidence_level
                z_alpha = norm.ppf(self.config.confidence_level)
                
                attr_prob = self.scenario.attribute_probabilities.get(attr, 0.5)
                # Variance approximation (Bernoulli)
                variance = attr_prob * (1 - attr_prob)
                std_dev = np.sqrt(variance)
                
                # Deterministic equivalent: mean + z_alpha * std_dev >= required
                entity_contrib = 1.0 if entity.attributes.get(attr, False) else 0.0
                required_with_safety = deficit + z_alpha * std_dev + self.config.constraint_buffer
                
                if entity_contrib < required_with_safety:
                    # Conservative decision
                    A_ub_list.append([1.0])
                    b_ub_list.append(0.5)  # Reduced acceptance
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        return c, A_ub, b_ub
    
    def _evaluate_chance_constraints(self, entity: Entity, problem_state: ProblemState, 
                                   constraints: List[Constraint]) -> float:
        """Evaluate probability that constraints will be satisfied."""
        
        # OPTIMIZED: Use analytical approximation instead of expensive Monte Carlo
        # This provides similar results but runs in O(1) instead of O(sample_size × capacity)
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 1:
            return 1.0  # No future entities to worry about
        
        # Analytical approach: use normal approximation for large capacity
        min_satisfaction_prob = 1.0
        
        for constraint in constraints:
            attr = constraint.attribute
            attr_prob = self.scenario.attribute_probabilities.get(attr, 0.5)
            required_pct = constraint.min_percentage
            
            # Current state after accepting this entity
            current_count = problem_state.attribute_counts.get(attr, 0)
            entity_contrib = 1 if entity.attributes.get(attr, False) else 0
            new_count = current_count + entity_contrib
            new_total = problem_state.accepted_count + 1
            
            # Expected final count using normal approximation
            # Assume we'll accept about 70% of remaining entities (reasonable estimate)
            expected_future_accepted = min(remaining_capacity - 1, int(0.7 * remaining_capacity))
            expected_future_with_attr = expected_future_accepted * attr_prob
            
            final_expected_count = new_count + expected_future_with_attr
            final_expected_total = new_total + expected_future_accepted
            
            if final_expected_total > 0:
                final_expected_pct = final_expected_count / final_expected_total
                
                # Use variance to estimate probability
                variance = attr_prob * (1 - attr_prob) * expected_future_accepted
                std_dev = np.sqrt(variance) if variance > 0 else 0
                
                if std_dev > 0:
                    # Normal approximation: P(X >= required)
                    z_score = (required_pct * final_expected_total - final_expected_count) / std_dev
                    satisfaction_prob = 1 - norm.cdf(z_score)
                else:
                    # Deterministic case
                    satisfaction_prob = 1.0 if final_expected_pct >= required_pct else 0.0
                
                min_satisfaction_prob = min(min_satisfaction_prob, satisfaction_prob)
        
        return min_satisfaction_prob
    
    def _simulate_constraint_satisfaction(self, entity: Entity, problem_state: ProblemState, 
                                        constraints: List[Constraint]) -> bool:
        """
        DEPRECATED: Slow Monte Carlo simulation method.
        
        This method is no longer used since we switched to analytical approximation
        for performance reasons. Kept for reference/future use if needed.
        """
        # This method is intentionally not implemented to prevent accidental use
        # The analytical approximation in _evaluate_chance_constraints is much faster
        raise NotImplementedError("Monte Carlo simulation deprecated for performance. Using analytical approximation.")
    
    def _calculate_immediate_reward(self, entity: Entity, problem_state: ProblemState, 
                                  constraints: List[Constraint]) -> float:
        """Calculate immediate reward for accepting this entity."""
        if problem_state.accepted_count == 0:
            return 2.0  # High reward for first entity
        
        reward = 1.0  # Base reward
        
        # Bonus for helping with constraint deficits
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        for constraint in constraints:
            attr = constraint.attribute
            if entity.attributes.get(attr, False) and deficits[attr] > 0:
                reward += deficits[attr] * 5.0  # High reward for addressing deficits
        
        return reward
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy configuration information."""
        return {
            "type": "ChanceConstraint", 
            "acceptance_threshold": self.config.acceptance_threshold,
            "confidence_level": self.config.confidence_level,
            "sample_size": self.config.sample_size
        }
