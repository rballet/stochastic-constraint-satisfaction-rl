from ..core.strategy_base import AbstractStrategy
from ..core.types import Entity, ProblemState, Constraint, Decision


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