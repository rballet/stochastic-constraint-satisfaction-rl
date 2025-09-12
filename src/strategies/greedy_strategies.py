from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

from ..core.strategy_base import AbstractStrategy
from ..core.types import Entity, ProblemState, Constraint, Decision


@dataclass
class WeightedGreedyConfig:
    """Configuration for Weighted Constraint Greedy strategy."""
    acceptance_threshold: float = 0.0 #0.0001 # Avoid accepting a no-benefit entity
    weight_strategy: str = "equal"  # "equal", "deficit_proportional", "constraint_difficulty"
    capacity_buffer: float = 0.0
    use_dilution_penalty: bool = True


@dataclass  
class AdaptiveGreedyConfig:
    """Configuration for Adaptive Threshold Greedy strategy."""
    base_threshold: float = 0.0
    capacity_pressure_factor: float = 0.2
    time_pressure_factor: float = 0.1
    capacity_buffer: float = 0.0


class GreedyStrategy(AbstractStrategy):
    """Greedy constraint-aware strategy."""
    
    def __init__(self):
        super().__init__("Greedy")
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: list[Constraint]) -> Decision:
        if problem_state.accepted_count == 0:
            return Decision.ACCEPT
        
        # Accept if entity helps with any unsatisfied constraint
        for constraint in constraints:
            attr = constraint.attribute
            current_count = problem_state.attribute_counts.get(attr, 0)
            current_percentage = current_count / problem_state.accepted_count
            
            if (current_percentage < constraint.min_percentage and 
                entity.attributes.get(attr, False)):
                return Decision.ACCEPT
        
        # If all constraints satisfied, accept to fill capacity
        all_satisfied = all(
            problem_state.get_attribute_percentage(c.attribute) >= c.min_percentage
            for c in constraints
        )
        
        return Decision.ACCEPT if all_satisfied else Decision.REJECT
    
    def _reset_internal_state(self) -> None:
        pass


class WeightedConstraintGreedy(AbstractStrategy):
    """Weighted Constraint Greedy strategy that balances multiple constraints using configurable weights."""
    
    def __init__(self, scenario=None, config: Optional[WeightedGreedyConfig] = None):
        super().__init__("WeightedGreedy")
        self.scenario = scenario
        self.config = config or WeightedGreedyConfig()
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        if problem_state.accepted_count == 0:
            return Decision.ACCEPT
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        
        # Check capacity with buffer
        if remaining_capacity <= self.config.capacity_buffer * problem_state.capacity:
            return Decision.REJECT
            
        # Calculate constraint weights
        weights = self._calculate_constraint_weights(problem_state, constraints)
        
        # Check if all constraints are already satisfied
        all_satisfied = all(
            problem_state.get_attribute_percentage(c.attribute) >= c.min_percentage
            for c in constraints
        )
        
        # If all constraints satisfied, accept to fill capacity (like basic Greedy)
        if all_satisfied:
            return Decision.ACCEPT
        
        # Calculate weighted benefit for this entity
        total_benefit = 0.0
        for i, constraint in enumerate(constraints):
            benefit = self._calculate_constraint_benefit(entity, constraint, problem_state)
            total_benefit += weights[i] * benefit
                    
        return Decision.ACCEPT if total_benefit > self.config.acceptance_threshold else Decision.REJECT
    
    def _calculate_constraint_weights(self, problem_state: ProblemState, constraints: List[Constraint]) -> List[float]:
        """Calculate weights for each constraint based on the configured strategy."""
        
        if self.config.weight_strategy == "equal":
            return [1.0 / len(constraints)] * len(constraints)
            
        elif self.config.weight_strategy == "deficit_proportional":
            deficits = []
            for constraint in constraints:
                current_count = problem_state.attribute_counts.get(constraint.attribute, 0)
                current_pct = current_count / max(problem_state.accepted_count, 1)
                deficit = max(0, constraint.min_percentage - current_pct)
                deficits.append(deficit)
            
            total_deficit = sum(deficits)
            if total_deficit == 0:
                return [1.0 / len(constraints)] * len(constraints)
            return [d / total_deficit for d in deficits]
            
        elif self.config.weight_strategy == "constraint_difficulty":
            weights = []
            for constraint in constraints:
                if self.scenario:
                    attr_prob = self.scenario.attribute_probabilities.get(constraint.attribute, 0.5)
                    # For infeasible constraints (prob < required), use higher weight
                    if attr_prob < constraint.min_percentage:
                        # Infeasible constraint gets high weight proportional to deficit
                        difficulty = constraint.min_percentage - attr_prob + 1.0
                    else:
                        # Feasible constraint gets weight based on how tight it is
                        difficulty = 1.0 / max(0.1, attr_prob - constraint.min_percentage)
                    weights.append(difficulty)
                else:
                    weights.append(1.0)  # Fallback to equal weights
            
            total_weight = sum(weights)
            if total_weight == 0:
                return [1.0 / len(constraints)] * len(constraints)
            return [w / total_weight for w in weights]
            
        else:
            return [1.0 / len(constraints)] * len(constraints)
    
    def _calculate_constraint_benefit(self, entity: Entity, constraint: Constraint, problem_state: ProblemState) -> float:
        """Calculate benefit of accepting entity for a specific constraint."""
        current_count = problem_state.attribute_counts.get(constraint.attribute, 0)
        current_pct = current_count / max(problem_state.accepted_count, 1)
        deficit = max(0, constraint.min_percentage - current_pct)
        
        if entity.attributes.get(constraint.attribute, False) and deficit > 0:
            # Positive benefit: helps close deficit
            return deficit
        elif self.config.use_dilution_penalty and not entity.attributes.get(constraint.attribute, False) and current_pct >= constraint.min_percentage:
            # Negative benefit: dilution effect
            return -current_count / (problem_state.accepted_count * (problem_state.accepted_count + 1))
        else:
            return 0.0
    
    def _reset_internal_state(self) -> None:
        pass


class AdaptiveThresholdGreedy(AbstractStrategy):
    """Adaptive Threshold Greedy strategy that dynamically adjusts acceptance threshold."""
    
    def __init__(self, config: Optional[AdaptiveGreedyConfig] = None):
        super().__init__("AdaptiveGreedy")
        self.config = config or AdaptiveGreedyConfig()
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        
        # Check capacity with buffer
        if remaining_capacity <= self.config.capacity_buffer * problem_state.capacity:
            return Decision.REJECT
            
        # Always accept first entity
        if problem_state.accepted_count == 0:
            return Decision.ACCEPT
            
        # Check if all constraints are already satisfied
        all_satisfied = all(
            problem_state.get_attribute_percentage(c.attribute) >= c.min_percentage
            for c in constraints
        )
        
        # If all constraints satisfied, accept to fill capacity
        if all_satisfied:
            return Decision.ACCEPT
        
        # Calculate adaptive threshold
        current_threshold = self._calculate_adaptive_threshold(problem_state)
        
        # Calculate entity benefit
        entity_benefit = self._calculate_entity_benefit(entity, problem_state, constraints)
        
        return Decision.ACCEPT if entity_benefit > current_threshold else Decision.REJECT
    
    def _calculate_adaptive_threshold(self, problem_state: ProblemState) -> float:
        """Calculate adaptive acceptance threshold based on capacity and time pressure."""
        
        # Capacity pressure factor (higher threshold when capacity is low)
        capacity_ratio = problem_state.accepted_count / problem_state.capacity
        capacity_pressure = 1 + self.config.capacity_pressure_factor * capacity_ratio
        
        # Time pressure factor (estimate based on decision count and capacity)
        # Assume we'll see roughly 2x capacity worth of entities
        estimated_total_decisions = problem_state.capacity * 2
        time_progress = min(1.0, self._decision_count / estimated_total_decisions)
        time_pressure = 1 - self.config.time_pressure_factor * time_progress
        
        return self.config.base_threshold * capacity_pressure * time_pressure
    
    def _calculate_entity_benefit(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> float:
        """Calculate total benefit of accepting this entity."""
        total_benefit = 0.0
        
        for constraint in constraints:
            current_count = problem_state.attribute_counts.get(constraint.attribute, 0)
            current_pct = current_count / max(problem_state.accepted_count, 1)
            deficit = max(0, constraint.min_percentage - current_pct)
            
            if entity.attributes.get(constraint.attribute, False) and deficit > 0:
                # Benefit from helping constraint
                total_benefit += deficit * 2.0  # Weight constraint help highly
            elif not entity.attributes.get(constraint.attribute, False) and deficit == 0:
                # Small penalty for dilution
                total_benefit -= 0.01
        
        # Base benefit for capacity utilization
        total_benefit += 0.05
        
        return total_benefit
    
    def _reset_internal_state(self) -> None:
        pass