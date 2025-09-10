"""
Simulation engine for stochastic constraint satisfaction problems.
"""

from typing import List, Optional, Dict, Any
import logging
from ..core.types import (
    Entity, ProblemState, Scenario, SimulationResult, Strategy, 
    EntityGenerator, Decision, ConstraintStatus
)


class SimulationEngine:
    """Main simulation engine for running constraint satisfaction scenarios."""
    
    def __init__(self, entity_generator: EntityGenerator, logger: Optional[logging.Logger] = None):
        self.entity_generator = entity_generator
        self.logger = logger or logging.getLogger(__name__)
        
    def run_simulation(
        self, 
        scenario: Scenario, 
        strategy: Strategy,
        seed: Optional[int] = None,
        log_decisions: bool = False
    ) -> SimulationResult:
        """Run a single simulation with given scenario and strategy."""
        
        # Reset components
        strategy.reset()
        self.entity_generator.reset(seed)
        
        # Initialize state
        problem_state = ProblemState()
        decision_log = [] if log_decisions else None
        entity_count = 0
        # Track empirical arrival stats irrespective of decisions
        arrival_total = 0
        arrival_attr_counts: Dict[str, int] = {}
        arrival_pair_counts: Dict[str, Dict[str, int]] = {}
        
        self.logger.info(f"Starting simulation: {scenario.name} with {strategy.name}")
        
        while not problem_state.is_terminal():
            entity_count += 1
            entity = self.entity_generator.generate_entity(scenario)
            entity.id = entity_count
            # Update arrival stats
            arrival_total += 1
            for a, v in entity.attributes.items():
                if v:
                    arrival_attr_counts[a] = arrival_attr_counts.get(a, 0) + 1
            # pair counts (off-diagonal only, canonical order)
            attrs_true = [a for a, v in entity.attributes.items() if v]
            for i in range(len(attrs_true)):
                for j in range(i + 1, len(attrs_true)):
                    ai, aj = attrs_true[i], attrs_true[j]
                    if ai > aj:
                        ai, aj = aj, ai
                    if ai not in arrival_pair_counts:
                        arrival_pair_counts[ai] = {}
                    arrival_pair_counts[ai][aj] = arrival_pair_counts[ai].get(aj, 0) + 1
            
            # Get strategy decision
            decision = strategy.decide(entity, problem_state, scenario.constraints)
            
            # Log decision if requested
            if log_decisions:
                decision_log.append({
                    'entity_id': entity.id,
                    'attributes': entity.attributes.copy(),
                    'decision': decision.value,
                    'problem_state': {
                        'accepted': problem_state.accepted_count,
                        'rejected': problem_state.rejected_count,
                        'attribute_counts': problem_state.attribute_counts.copy()
                    }
                })
            
            # Update problem state
            if decision == Decision.ACCEPT:
                problem_state.accepted_count += 1
                for attr, value in entity.attributes.items():
                    if value:  # Only count positive attributes
                        problem_state.attribute_counts[attr] = (
                            problem_state.attribute_counts.get(attr, 0) + 1
                        )
            else:
                problem_state.rejected_count += 1
            
            # Check for early termination conditions
            if problem_state.is_terminal():
                break
        
        # Calculate final results
        constraints_satisfied = self._check_all_constraints_satisfied(scenario.constraints, problem_state)
        final_percentages = {
            attr: problem_state.get_attribute_percentage(attr) 
            for attr in scenario.attributes
        }
        
        result = SimulationResult(
            scenario_name=scenario.name,
            strategy_name=strategy.name,
            accepted_count=problem_state.accepted_count,
            rejected_count=problem_state.rejected_count,
            constraints_satisfied=constraints_satisfied,
            final_attribute_percentages=final_percentages,
            decision_log=decision_log,
            arrival_total=arrival_total,
            arrival_attribute_counts=arrival_attr_counts,
            arrival_pair_counts=arrival_pair_counts
        )
        
        self.logger.info(f"Simulation complete: {result.success}, "
                        f"Accepted: {result.accepted_count}, "
                        f"Rejected: {result.rejected_count}")
        
        return result
    
    def run_multiple_simulations(
        self,
        scenario: Scenario,
        strategy: Strategy,
        num_runs: int = 100,
        seeds: Optional[List[int]] = None,
        log_decisions: bool = False
    ) -> List[SimulationResult]:
        """Run multiple simulations and return results."""
        
        if seeds is None:
            seeds = list(range(num_runs))
        elif len(seeds) != num_runs:
            raise ValueError("Number of seeds must match number of runs")
        
        results = []
        for i, seed in enumerate(seeds):
            self.logger.debug(f"Running simulation {i+1}/{num_runs}")
            result = self.run_simulation(scenario, strategy, seed, log_decisions)
            results.append(result)
        
        return results
    
    def _check_all_constraints_satisfied(
        self, 
        constraints: List, 
        problem_state: ProblemState
    ) -> bool:
        """Check if all constraints are satisfied."""
        for constraint in constraints:
            if not constraint.is_satisfied(
                problem_state.accepted_count,
                problem_state.attribute_counts.get(constraint.attribute, 0)
            ):
                return False
        return True
    
    def get_constraint_status(
        self, 
        constraints: List, 
        problem_state: ProblemState
    ) -> List[ConstraintStatus]:
        """Get current status of all constraints."""
        return [
            ConstraintStatus.from_constraint(constraint, problem_state)
            for constraint in constraints
        ]


class DetailedSimulationEngine(SimulationEngine):
    """Extended simulation engine with detailed logging and analysis."""
    
    def __init__(self, entity_generator: EntityGenerator, logger: Optional[logging.Logger] = None):
        super().__init__(entity_generator, logger)
        self.step_callbacks = []
        
    def add_step_callback(self, callback):
        """Add a callback function called after each decision."""
        self.step_callbacks.append(callback)
        
    def run_simulation(
        self, 
        scenario: Scenario, 
        strategy: Strategy,
        seed: Optional[int] = None,
        log_decisions: bool = True  # Default to True for detailed engine
    ) -> SimulationResult:
        """Run simulation with detailed callbacks."""
        
        # Call parent implementation but with enhanced logging
        strategy.reset()
        self.entity_generator.reset(seed)
        
        problem_state = ProblemState()
        decision_log = [] if log_decisions else None
        entity_count = 0
        
        while not problem_state.is_terminal():
            entity_count += 1
            entity = self.entity_generator.generate_entity(scenario)
            entity.id = entity_count
            
            # Get constraint status before decision
            constraint_status = self.get_constraint_status(scenario.constraints, problem_state)
            
            decision = strategy.decide(entity, problem_state, scenario.constraints)
            
            # Log detailed decision info
            if log_decisions:
                decision_info = {
                    'entity_id': entity.id,
                    'attributes': entity.attributes.copy(),
                    'decision': decision.value,
                    'problem_state_before': {
                        'accepted': problem_state.accepted_count,
                        'rejected': problem_state.rejected_count,
                        'attribute_counts': problem_state.attribute_counts.copy(),
                        'attribute_percentages': {
                            attr: problem_state.get_attribute_percentage(attr)
                            for attr in scenario.attributes
                        }
                    },
                    'constraint_status': [
                        {
                            'attribute': cs.constraint.attribute,
                            'required_pct': cs.required_percentage,
                            'current_pct': cs.current_percentage,
                            'satisfied': cs.is_satisfied,
                            'urgency': cs.urgency_score
                        }
                        for cs in constraint_status
                    ]
                }
                decision_log.append(decision_info)
            
            # Update state
            if decision == Decision.ACCEPT:
                problem_state.accepted_count += 1
                for attr, value in entity.attributes.items():
                    if value:
                        problem_state.attribute_counts[attr] = (
                            problem_state.attribute_counts.get(attr, 0) + 1
                        )
            else:
                problem_state.rejected_count += 1
            
            # Call step callbacks
            for callback in self.step_callbacks:
                callback(entity, decision, problem_state, constraint_status)
        
        # Build result
        constraints_satisfied = self._check_all_constraints_satisfied(scenario.constraints, problem_state)
        final_percentages = {
            attr: problem_state.get_attribute_percentage(attr) 
            for attr in scenario.attributes
        }
        
        return SimulationResult(
            scenario_name=scenario.name,
            strategy_name=strategy.name,
            accepted_count=problem_state.accepted_count,
            rejected_count=problem_state.rejected_count,
            constraints_satisfied=constraints_satisfied,
            final_attribute_percentages=final_percentages,
            decision_log=decision_log
        )