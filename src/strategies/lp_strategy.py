"""
Linear Programming strategy for stochastic constraint satisfaction problems.

This strategy uses linear programming to make optimal decisions based on:
1. Current problem state
2. Remaining capacity
3. Constraint requirements
4. Expected future arrivals (based on scenario distributions)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass

from ..core.strategy_base import AbstractStrategy
from ..core.types import Entity, ProblemState, Constraint, Decision, Scenario


@dataclass
class LPStrategyConfig:
    """Configuration for LP strategy."""
    acceptance_threshold: float = 0.5  # Accept if LP probability > threshold
    lookahead_horizon: int = 50  # Number of future entities to consider
    use_conservative_estimates: bool = True  # Use conservative probability estimates
    constraint_buffer: float = 0.05  # Buffer above minimum constraint requirements


class LinearProgrammingStrategy(AbstractStrategy):
    """
    Generic Linear Programming strategy for constraint satisfaction.
    
    The strategy formulates an LP to maximize expected utility while
    satisfying constraints, considering both current state and future arrivals.
    """
    
    def __init__(self, scenario: Scenario, config: Optional[LPStrategyConfig] = None):
        super().__init__("LinearProgramming")
        self.scenario = scenario
        self.config = config or LPStrategyConfig()
        
        # Cache for optimization
        self._attribute_to_index = {attr: i for i, attr in enumerate(scenario.attributes)}
        self._constraint_attributes = {c.attribute for c in scenario.constraints}
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using linear programming optimization."""
        
        # If we're close to capacity limit, be more selective
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 1:
            return self._make_final_decisions(entity, problem_state, constraints)
        
        # Formulate and solve LP
        acceptance_probability = self._solve_lp_for_entity(entity, problem_state, constraints, remaining_capacity)
        
        # Make decision based on probability threshold
        return Decision.ACCEPT if acceptance_probability >= self.config.acceptance_threshold else Decision.REJECT
    
    def _solve_lp_for_entity(
        self, 
        entity: Entity, 
        problem_state: ProblemState, 
        constraints: List[Constraint],
        remaining_capacity: int
    ) -> float:
        """Solve LP to determine optimal acceptance probability for current entity."""
        
        # Current state
        current_accepted = problem_state.accepted_count
        current_attr_counts = problem_state.attribute_counts.copy()
        
        # Entity attributes as binary vector
        entity_vector = self._entity_to_vector(entity)
        
        # If accepting this entity would violate capacity, reject
        if remaining_capacity <= 0:
            return 0.0
        
        # Check if accepting this entity helps with constraints
        utility_score = self._calculate_entity_utility(entity, current_accepted, current_attr_counts, constraints)
        
        # Consider future arrivals using scenario probabilities
        future_utility = self._estimate_future_utility(
            entity_vector, current_accepted + 1, current_attr_counts, 
            constraints, remaining_capacity - 1
        )
        
        # Combine immediate and future utility
        total_utility = utility_score + future_utility
        
        # Convert utility to probability (sigmoid-like function)
        probability = self._utility_to_probability(total_utility)
        
        return probability
    
    def _calculate_entity_utility(
        self, 
        entity: Entity, 
        current_accepted: int, 
        current_attr_counts: Dict[str, int],
        constraints: List[Constraint]
    ) -> float:
        """Calculate utility of accepting this specific entity."""
        
        if current_accepted == 0:
            return 1.0  # Always good to start accepting
        
        utility = 0.0
        
        # Check each constraint
        for constraint in constraints:
            attr = constraint.attribute
            current_count = current_attr_counts.get(attr, 0)
            current_percentage = current_count / current_accepted
            
            # If entity has this attribute
            if entity.attributes.get(attr, False):
                # Calculate new percentage if we accept
                new_percentage = (current_count + 1) / (current_accepted + 1)
                
                # Utility is higher if this helps meet constraint
                if current_percentage < constraint.min_percentage:
                    # How much closer to constraint does this get us?
                    improvement = new_percentage - current_percentage
                    utility += improvement * 10.0  # Weight constraint satisfaction highly
                else:
                    # Already meeting constraint, small positive utility
                    utility += 0.1
            else:
                # Entity doesn't have this attribute
                new_percentage = current_count / (current_accepted + 1)
                
                # Check if this moves us away from constraint
                if current_percentage >= constraint.min_percentage:
                    # We can afford some dilution
                    if new_percentage >= constraint.min_percentage - self.config.constraint_buffer:
                        utility += 0.0  # Neutral
                    else:
                        utility -= 5.0  # Penalty for violating constraint
                else:
                    # Already below constraint, this makes it worse
                    utility -= 2.0
        
        return utility
    
    def _estimate_future_utility(
        self,
        current_entity_vector: np.ndarray,
        projected_accepted: int,
        projected_attr_counts: Dict[str, int],
        constraints: List[Constraint],
        remaining_capacity: int
    ) -> float:
        """Estimate utility from future optimal decisions."""
        
        if remaining_capacity <= 0:
            return 0.0
        
        # Update projected state if we accept current entity
        updated_attr_counts = projected_attr_counts.copy()
        for i, attr in enumerate(self.scenario.attributes):
            if current_entity_vector[i] > 0:
                updated_attr_counts[attr] = updated_attr_counts.get(attr, 0) + 1
        
        # Formulate LP for remaining decisions
        # Variables: x_a for each attribute combination a
        # Objective: maximize utility subject to constraints
        
        # Simplification: estimate based on constraint gaps
        future_utility = 0.0
        entities_to_consider = min(self.config.lookahead_horizon, remaining_capacity)
        
        for constraint in constraints:
            attr = constraint.attribute
            current_count = updated_attr_counts.get(attr, 0)
            current_percentage = current_count / projected_accepted if projected_accepted > 0 else 0
            
            if current_percentage < constraint.min_percentage:
                # Need more of this attribute
                required_additional = max(0, 
                    int(constraint.min_percentage * (projected_accepted + entities_to_consider)) - current_count
                )
                
                # Probability of getting required attributes from future arrivals
                attr_prob = self.scenario.attribute_probabilities.get(attr, 0.5)
                expected_additional = entities_to_consider * attr_prob
                
                if expected_additional >= required_additional:
                    future_utility += 1.0  # We can likely meet this constraint
                else:
                    future_utility -= (required_additional - expected_additional) * 0.5
        
        return future_utility / len(constraints)  # Normalize
    
    def _utility_to_probability(self, utility: float) -> float:
        """Convert utility score to acceptance probability."""
        # Sigmoid-like function to map utility to [0, 1]
        return 1.0 / (1.0 + np.exp(-utility))
    
    def _make_final_decisions(
        self, 
        entity: Entity, 
        problem_state: ProblemState, 
        constraints: List[Constraint]
    ) -> Decision:
        """Make decisions when near capacity limit."""
        
        # Check if accepting this entity improves constraint satisfaction
        improvement_score = 0.0
        
        for constraint in constraints:
            attr = constraint.attribute
            current_count = problem_state.attribute_counts.get(attr, 0)
            current_percentage = current_count / problem_state.accepted_count if problem_state.accepted_count > 0 else 0
            
            if current_percentage < constraint.min_percentage and entity.attributes.get(attr, False):
                # This entity helps with an unsatisfied constraint
                improvement_score += constraint.min_percentage - current_percentage
        
        # Accept if this entity significantly improves constraint satisfaction
        return Decision.ACCEPT if improvement_score > 0.1 else Decision.REJECT
    
    def _entity_to_vector(self, entity: Entity) -> np.ndarray:
        """Convert entity to binary vector representation."""
        vector = np.zeros(len(self.scenario.attributes))
        for i, attr in enumerate(self.scenario.attributes):
            vector[i] = 1.0 if entity.attributes.get(attr, False) else 0.0
        return vector
    
    def _reset_internal_state(self) -> None:
        """Reset any internal state."""
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy configuration information."""
        return {
            "type": "LinearProgramming",
            "acceptance_threshold": self.config.acceptance_threshold,
            "lookahead_horizon": self.config.lookahead_horizon,
            "constraint_buffer": self.config.constraint_buffer,
            "attributes": self.scenario.attributes,
            "constraints": [c.description for c in self.scenario.constraints]
        }
