"""
Core types and data structures for stochastic constraint satisfaction problems.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Any
from enum import Enum
import abc


class Decision(Enum):
    """Decision enum for agent choices."""
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class Entity:
    """Represents an entity in a stochastic constraint satisfaction problem."""
    attributes: Dict[str, bool]
    id: Optional[int] = None
    
    def has_attribute(self, attribute: str) -> bool:
        """Check if entity has a specific attribute."""
        return self.attributes.get(attribute, False)


@dataclass
class Constraint:
    """Represents a constraint that must be satisfied."""
    attribute: str
    min_percentage: float
    description: str = ""
    
    def is_satisfied(self, total_count: int, attribute_count: int) -> bool:
        """Check if constraint is currently satisfied."""
        if total_count == 0:
            return False  # Empty state cannot satisfy any constraint
        return (attribute_count / total_count) >= self.min_percentage


@dataclass
class Scenario:
    """Represents a problem scenario."""
    name: str
    attributes: List[str]
    constraints: List[Constraint]
    attribute_probabilities: Dict[str, float]
    capacity: int
    max_rejections: int
    attribute_correlations: Dict[tuple, float] = field(default_factory=dict)
    category: str = "DEFAULT"
    description: str = ""


@dataclass
class ProblemState:
    """Current state of the constraint satisfaction problem."""
    accepted_count: int = 0
    rejected_count: int = 0
    attribute_counts: Dict[str, int] = field(default_factory=dict)
    scenario: Scenario = field(default_factory=Scenario)

    @property
    def capacity(self) -> int:
        """Get capacity from scenario."""
        return self.scenario.capacity

    @property
    def max_rejections(self) -> int:
        """Get max rejections from scenario."""
        return self.scenario.max_rejections

    def is_full(self) -> bool:
        """Check if capacity is reached."""
        return self.accepted_count >= self.capacity
    
    def max_rejections_reached(self) -> bool:
        """Check if maximum rejections reached."""
        return self.rejected_count >= self.max_rejections
    
    def is_terminal(self) -> bool:
        """Check if problem is in terminal state."""
        return self.is_full() or self.max_rejections_reached()
    
    def get_attribute_percentage(self, attribute: str) -> float:
        """Get current percentage of entities with given attribute."""
        if self.accepted_count == 0:
            return 0.0
        return self.attribute_counts.get(attribute, 0) / self.accepted_count

    def is_satisfied(self) -> bool:
        """Check if all constraints are satisfied."""
        return all(self.get_attribute_percentage(c.attribute) >= c.min_percentage for c in self.scenario.constraints)

    def is_success(self) -> bool:
        """Check if problem is successful."""
        return self.is_satisfied() and self.is_full()


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    scenario_name: str
    strategy_name: str
    accepted_count: int
    rejected_count: int
    constraints_satisfied: bool
    final_attribute_percentages: Dict[str, float]
    decision_log: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    # Empirical arrival statistics for generated entities (independent of decisions)
    arrival_total: int = 0
    arrival_attribute_counts: Dict[str, int] = field(default_factory=dict)
    arrival_pair_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)



class Strategy(Protocol):
    """Protocol for strategy implementations."""
    
    @abc.abstractmethod
    def decide(self, entity: Entity, state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make a decision whether to accept or reject an entity."""
        pass
    
    @abc.abstractmethod
    def reset(self) -> None:
        """Reset strategy state for a new simulation."""
        pass
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class EntityGenerator(Protocol):
    """Protocol for entity generation."""
    
    @abc.abstractmethod
    def generate_entity(self, scenario: Scenario) -> Entity:
        """Generate an entity according to scenario distributions."""
        pass
    
    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset generator state."""
        pass


@dataclass
class ConstraintStatus:
    """Status of a constraint at a given time."""
    constraint: Constraint
    current_percentage: float
    required_percentage: float
    is_satisfied: bool
    entities_needed: int
    urgency_score: float
    
    @classmethod
    def from_constraint(cls, constraint: Constraint, state: ProblemState) -> 'ConstraintStatus':
        """Create constraint status from current problem state."""
        current_pct = state.get_attribute_percentage(constraint.attribute)
        is_satisfied = constraint.is_satisfied(
            state.accepted_count,
            state.attribute_counts.get(constraint.attribute, 0)
        )
        
        # Calculate entities needed to satisfy constraint
        entities_needed = 0
        if not is_satisfied:
            total_needed = constraint.min_percentage * state.capacity()
            current_count = state.attribute_counts.get(constraint.attribute, 0)
            entities_needed = max(0, int(total_needed - current_count))
        
        # Calculate urgency (higher = more urgent)
        remaining_capacity = state.capacity() - state.accepted_count
        urgency_score = 0.0
        if remaining_capacity > 0 and entities_needed > 0:
            urgency_score = entities_needed / remaining_capacity
        
        return cls(
            constraint=constraint,
            current_percentage=current_pct,
            required_percentage=constraint.min_percentage,
            is_satisfied=is_satisfied,
            entities_needed=entities_needed,
            urgency_score=urgency_score
        )