"""
Abstract base class for strategies in stochastic constraint satisfaction problems.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .types import Entity, ProblemState, Constraint, Decision


class AbstractStrategy(ABC):
    """Abstract base class for all strategies."""
    
    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._decision_count = 0
        self._state = {}
        
    @property
    def name(self) -> str:
        """Return strategy name."""
        return self._name
    
    @property
    def decision_count(self) -> int:
        """Return number of decisions made."""
        return self._decision_count
    
    @property
    def state(self) -> Dict[str, Any]:
        """Return internal strategy state."""
        return self._state.copy()
    
    def decide(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """
        Make a decision whether to accept or reject an entity.
        
        This method includes automatic optimization: once all constraints are satisfied,
        it will accept all remaining entities to maximize capacity.
        
        Args:
            entity: The entity to make a decision about
            problem_state: Current state of the problem
            constraints: List of constraints that must be satisfied
            
        Returns:
            Decision.ACCEPT or Decision.REJECT
        """
        # If all constraints are satisfied, accept everyone remaining
        if self.should_accept_all_remaining(problem_state, constraints):
            return Decision.ACCEPT
        
        # Otherwise delegate to strategy-specific logic
        return self._decide_impl(entity, problem_state, constraints)
    
    @abstractmethod
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """
        Strategy-specific decision logic.
        
        This method should contain the core decision-making logic for each strategy.
        The optimization is automatically handled by the base decide() method.
        
        Args:
            entity: The entity to make a decision about
            problem_state: Current state of the problem
            constraints: List of constraints that must be satisfied
            
        Returns:
            Decision.ACCEPT or Decision.REJECT
        """
        pass
    
    def reset(self) -> None:
        """Reset strategy state for a new simulation."""
        self._decision_count = 0
        self._state.clear()
        self._optimization_activated = False
        self._reset_internal_state()
    
    @abstractmethod
    def _reset_internal_state(self) -> None:
        """Reset any strategy-specific internal state."""
        pass
    
    def _increment_decision_count(self) -> None:
        """Increment the decision counter. Call this in decide() implementations."""
        self._decision_count += 1
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Return information about the strategy configuration.
        Override this to provide strategy-specific information.
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "decisions_made": self.decision_count
        }
    
    def are_all_constraints_satisfied(self, problem_state: ProblemState, constraints: List[Constraint]) -> bool:
        """
        Check if all constraints are currently satisfied.
        
        This is useful for optimization: once all constraints are met,
        we can accept everyone else to fill to capacity.
        
        Args:
            problem_state: Current state of the problem
            constraints: List of constraints to check
            
        Returns:
            True if all constraints are satisfied, False otherwise
        """
        for constraint in constraints:
            current_count = problem_state.attribute_counts.get(constraint.attribute, 0)
            if not constraint.is_satisfied(problem_state.accepted_count, current_count):
                return False
        return True
    
    def should_accept_all_remaining(self, problem_state: ProblemState, constraints: List[Constraint]) -> bool:
        """
        Check if we should accept all remaining entities.
        
        This returns True when:
        1. All constraints are satisfied in terms of percentages
        2. We have enough absolute counts to meet the intended minimums
        3. There's still capacity in the problem
        
        The second condition prevents premature optimization when we have good percentages
        but insufficient absolute counts to meet the real requirements.
        
        Args:
            problem_state: Current state of the problem
            constraints: List of constraints to check
            
        Returns:
            True if we should accept all remaining entities, False otherwise
        """
        if problem_state.accepted_count >= problem_state.capacity:
            return False
            
        # Check if all percentage constraints are satisfied
        if not self.are_all_constraints_satisfied(problem_state, constraints):
            return False
            
        # Additional check: ensure we have enough absolute counts
        # Assumes constraint percentages are for full capacity
        for constraint in constraints:
            current_count = problem_state.attribute_counts.get(constraint.attribute, 0)
            required_absolute_count = constraint.min_percentage * problem_state.capacity
            if current_count < required_absolute_count:
                return False
                
        return True
    
    def on_problem_start(self, constraints: List[Constraint], capacity: int) -> None:
        """
        Called at the start of each problem. Override to perform initialization.
        
        Args:
            constraints: The constraints for this problem
            capacity: Maximum capacity
        """
        pass
    
    def on_problem_end(self, problem_state: ProblemState, success: bool) -> None:
        """
        Called at the end of each problem. Override to perform cleanup or learning.
        
        Args:
            problem_state: Final state of the problem
            success: Whether the problem was successful (constraints satisfied)
        """
        pass


class RandomStrategy(AbstractStrategy):
    """Random strategy for baseline comparison."""
    
    def __init__(self, acceptance_rate: float = 0.7, seed: Optional[int] = None):
        super().__init__()
        self.acceptance_rate = acceptance_rate
        self.seed = seed
        self._random_state = None
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make random decision based on acceptance rate."""
        self._increment_decision_count()
        
        if self._random_state is None:
            import random
            self._random_state = random.Random(self.seed)
        
        return Decision.ACCEPT if self._random_state.random() < self.acceptance_rate else Decision.REJECT
    
    def _reset_internal_state(self) -> None:
        """Reset random state."""
        if self.seed is not None:
            import random
            self._random_state = random.Random(self.seed)
        else:
            self._random_state = None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information."""
        info = super().get_strategy_info()
        info.update({
            "acceptance_rate": self.acceptance_rate,
            "seed": self.seed
        })
        return info


class AlwaysAcceptStrategy(AbstractStrategy):
    """Strategy that always accepts (for testing)."""
    
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Always accept."""
        self._increment_decision_count()
        return Decision.ACCEPT
    
    def _reset_internal_state(self) -> None:
        """No internal state to reset."""
        pass


class AlwaysRejectStrategy(AbstractStrategy):
    """Strategy that always rejects (for testing edge cases)."""
    
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Always reject."""
        self._increment_decision_count()
        return Decision.REJECT
    
    def _reset_internal_state(self) -> None:
        """No internal state to reset."""
        pass


class AcceptFirstNStrategy(AbstractStrategy):
    """Strategy that accepts first N entities, then rejects all (for testing)."""
    
    def __init__(self, n: int = 1000):
        super().__init__()
        self.n = n
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Accept first N entities."""
        self._increment_decision_count()
        return Decision.ACCEPT if problem_state.accepted_count < self.n else Decision.REJECT
    
    def _reset_internal_state(self) -> None:
        """No internal state to reset."""
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information."""
        info = super().get_strategy_info()
        info["accept_count"] = self.n
        return info