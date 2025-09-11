"""
Adaptive Linear Programming strategy with primal-dual pricing and dynamic learning.

This module implements an enhanced LP approach that uses dual variables as "shadow prices"
to make intelligent decisions even when the primal LP becomes infeasible.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass
from collections import deque

from .lp_strategy import LinearProgrammingStrategy, BaseLPStrategy, LPStrategyConfig
from ..core.types import Entity, ProblemState, Constraint, Decision, Scenario


@dataclass
class AdaptiveLPConfig(LPStrategyConfig):
    """Configuration for Adaptive LP strategy with primal-dual features."""
    
    # Primal-Dual parameters
    dual_memory_length: int = 50  # How many dual solutions to remember
    price_smoothing_factor: float = 0.3  # EMA smoothing for shadow prices
    infeasibility_penalty: float = 100.0  # Penalty for constraint violations
    
    # Adaptive parameters
    learning_rate: float = 0.1  # Rate for updating acceptance thresholds
    min_acceptance_threshold: float = 0.1  # Minimum threshold to prevent over-rejection
    max_acceptance_threshold: float = 0.9  # Maximum threshold to ensure some acceptance
    
    # Rolling horizon parameters
    adaptive_horizon: bool = True  # Whether to adapt horizon based on constraint tightness
    min_horizon: int = 10  # Minimum lookahead horizon
    max_horizon: int = 100  # Maximum lookahead horizon
    
    # Feasibility recovery parameters
    feasibility_tolerance: float = 1e-6  # Tolerance for constraint satisfaction
    use_relaxed_constraints: bool = True  # Whether to use constraint relaxation


class AdaptiveLPStrategy(LinearProgrammingStrategy):
    """
    Adaptive Linear Programming strategy that learns constraint prices and adapts to infeasibility.
    
    Key features:
    1. Uses dual variables as shadow prices for constraint importance
    2. Adapts acceptance thresholds based on constraint tightness
    3. Maintains memory of constraint difficulty patterns
    4. Handles infeasibility through intelligent fallback based on prices
    """
    
    def __init__(self, scenario: Scenario, config: Optional[AdaptiveLPConfig] = None):
        super().__init__(scenario, config or AdaptiveLPConfig())
        self._name = "AdaptiveLP"
        self.config: AdaptiveLPConfig = self.config
        
        # Primal-Dual state
        self.dual_memory: deque = deque(maxlen=self.config.dual_memory_length)
        self.shadow_prices: Dict[str, float] = {}  # Current shadow prices for each constraint
        self.constraint_tightness: Dict[str, float] = {}  # How often each constraint is tight
        
        # Adaptive state
        self.current_acceptance_threshold = self.config.acceptance_threshold
        self.constraint_violation_history: Dict[str, deque] = {}
        self.infeasibility_count = 0
        self.total_decisions = 0
        
        # Initialize tracking for each constraint
        for constraint in scenario.constraints:
            self.shadow_prices[constraint.attribute] = 0.0
            self.constraint_tightness[constraint.attribute] = 0.0
            self.constraint_violation_history[constraint.attribute] = deque(maxlen=20)
    
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using adaptive LP with dual pricing."""
        
        self.total_decisions += 1
        
        remaining_capacity = problem_state.capacity - problem_state.accepted_count
        if remaining_capacity <= 0:
            return Decision.REJECT
        
        # Try to solve LP with current formulation
        try:
            c, A_ub, b_ub = self._formulate_adaptive_lp(entity, problem_state, constraints)
            if c is None:
                return self._price_based_decision(entity, problem_state, constraints)
            
            bounds = [(0, 1), (0, remaining_capacity)]
            result = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.config.solver_method)
            
            if result.success:
                # Extract dual variables (shadow prices) from successful LP solution
                self._update_dual_prices(result, constraints)
                
                x_t = result.x[0]
                decision = Decision.ACCEPT if x_t >= self.current_acceptance_threshold else Decision.REJECT
                
                # Update adaptive parameters based on success
                self._update_adaptive_parameters(True, entity, problem_state, constraints)
                
                return decision
                
            else:
                # LP is infeasible - use price-based decision
                self.infeasibility_count += 1
                self._update_adaptive_parameters(False, entity, problem_state, constraints)
                return self._price_based_decision(entity, problem_state, constraints)
                
        except Exception:
            # Solver error - fallback to price-based decision
            return self._price_based_decision(entity, problem_state, constraints)
    
    def _formulate_adaptive_lp(self, entity: Entity, problem_state: ProblemState, 
                              constraints: List[Constraint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Formulate LP with adaptive horizon and constraint relaxation if needed.
        """
        
        n_t = problem_state.accepted_count
        remaining_capacity = problem_state.capacity - n_t
        
        if remaining_capacity <= self.config.min_capacity_for_lp:
            return None, None, None
        
        # Adaptive horizon based on constraint tightness
        if self.config.adaptive_horizon:
            avg_tightness = np.mean(list(self.constraint_tightness.values())) if self.constraint_tightness else 0.5
            horizon_factor = 1.0 + avg_tightness  # More lookahead when constraints are tight
            adaptive_horizon = int(self.config.lookahead_horizon * horizon_factor)
            horizon = min(max(adaptive_horizon, self.config.min_horizon), 
                         self.config.max_horizon, remaining_capacity - 1)
        else:
            horizon = min(self.config.lookahead_horizon, remaining_capacity - 1)
        
        # Simplified 2-variable formulation: [x_t, y_future]
        num_vars = 2
        
        # Enhanced objective with dual price information
        c = np.zeros(num_vars)
        c[0] = self._calculate_dual_aware_reward(entity, problem_state, constraints)
        c[1] = self.config.future_reward_discount * self.config.base_reward
        
        # Constraints
        A_ub_list = []
        b_ub_list = []
        
        # Capacity constraint
        capacity_constraint = np.array([1.0, 1.0])
        A_ub_list.append(capacity_constraint)
        b_ub_list.append(remaining_capacity)
        
        # Attribute constraints with potential relaxation
        for constraint in constraints:
            attr = constraint.attribute
            theta_j = constraint.min_percentage
            p_j = self.scenario.attribute_probabilities.get(attr, 0.5)
            entity_has_attr = 1.0 if entity.attributes.get(attr, False) else 0.0
            c_j_t = problem_state.attribute_counts.get(attr, 0)
            
            # Calculate constraint coefficients
            rhs = theta_j * n_t - c_j_t
            x_t_coeff = entity_has_attr - theta_j
            y_future_coeff = p_j - theta_j
            
            # Apply constraint relaxation if this constraint has been problematic
            shadow_price = self.shadow_prices.get(attr, 0.0)
            if self.config.use_relaxed_constraints and shadow_price > self.config.infeasibility_penalty:
                # Relax this constraint by reducing its requirement
                relaxation_factor = 0.9  # Relax by 10%
                rhs *= relaxation_factor
            
            # Convert to LP standard form
            attr_constraint = np.array([-x_t_coeff, -y_future_coeff])
            bound_value = -rhs + self.config.constraint_buffer
            
            A_ub_list.append(attr_constraint)
            b_ub_list.append(bound_value)
        
        # Convert to numpy arrays
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        return c, A_ub, b_ub
    
    def _calculate_dual_aware_reward(self, entity: Entity, problem_state: ProblemState, 
                                   constraints: List[Constraint]) -> float:
        """Calculate reward incorporating dual price information."""
        
        base_reward = self._calculate_immediate_reward(entity, problem_state, constraints)
        
        # Add dual price bonuses/penalties
        dual_adjustment = 0.0
        for constraint in constraints:
            attr = constraint.attribute
            entity_has_attr = entity.attributes.get(attr, False)
            shadow_price = self.shadow_prices.get(attr, 0.0)
            
            if entity_has_attr:
                # Bonus for helping with expensive constraints
                dual_adjustment += shadow_price * 0.1
            else:
                # Penalty for not helping with expensive constraints
                dual_adjustment -= shadow_price * 0.05
        
        return base_reward + dual_adjustment
    
    def _calculate_immediate_reward(self, entity: Entity, problem_state: ProblemState, 
                                  constraints: List[Constraint]) -> float:
        """Calculate immediate reward - delegate to parent class."""
        return super()._calculate_immediate_reward(entity, problem_state, constraints)
    
    def _price_based_decision(self, entity: Entity, problem_state: ProblemState, 
                            constraints: List[Constraint]) -> Decision:
        """Make decision based on shadow prices when LP is infeasible."""
        
        # Calculate value based on constraint shadow prices
        total_value = 0.0
        total_penalty = 0.0
        
        deficits = self._calculate_constraint_deficit(problem_state, constraints)
        
        for constraint in constraints:
            attr = constraint.attribute
            entity_has_attr = entity.attributes.get(attr, False)
            shadow_price = self.shadow_prices.get(attr, 0.0)
            deficit = deficits.get(attr, 0.0)
            
            if entity_has_attr:
                # Value for helping constraint, scaled by shadow price and deficit
                constraint_value = shadow_price * (1.0 + deficit)
                total_value += constraint_value
            else:
                # Penalty for not helping, especially if constraint is expensive
                constraint_penalty = shadow_price * deficit * 0.5
                total_penalty += constraint_penalty
        
        net_value = total_value - total_penalty
        
        # Decision based on net value compared to adaptive threshold
        decision_threshold = self.current_acceptance_threshold * max(1.0, np.mean(list(self.shadow_prices.values())))
        
        return Decision.ACCEPT if net_value > decision_threshold else Decision.REJECT
    
    def _update_dual_prices(self, lp_result, constraints: List[Constraint]):
        """Update shadow prices using dual variables from LP solution."""
        
        if not hasattr(lp_result, 'slack') or lp_result.slack is None:
            return
        
        # Extract constraint shadow prices from slack variables
        # In scipy.optimize.linprog, tight constraints (slack ≈ 0) have high shadow prices
        slacks = lp_result.slack
        
        for i, constraint in enumerate(constraints):
            attr = constraint.attribute
            constraint_idx = i + 1  # +1 because first constraint is capacity
            
            if constraint_idx < len(slacks):
                # Inverse relationship: small slack = high shadow price
                raw_price = 1.0 / (slacks[constraint_idx] + self.config.feasibility_tolerance)
                
                # Smooth the price using exponential moving average
                if attr in self.shadow_prices:
                    self.shadow_prices[attr] = (
                        self.config.price_smoothing_factor * raw_price + 
                        (1 - self.config.price_smoothing_factor) * self.shadow_prices[attr]
                    )
                else:
                    self.shadow_prices[attr] = raw_price
                
                # Update constraint tightness (how often constraint is near binding)
                is_tight = slacks[constraint_idx] < self.config.feasibility_tolerance * 10
                self.constraint_tightness[attr] = (
                    self.config.price_smoothing_factor * (1.0 if is_tight else 0.0) + 
                    (1 - self.config.price_smoothing_factor) * self.constraint_tightness.get(attr, 0.0)
                )
        
        # Store dual solution in memory
        dual_info = {
            'prices': self.shadow_prices.copy(),
            'tightness': self.constraint_tightness.copy(),
            'objective': -lp_result.fun if lp_result.fun is not None else 0.0
        }
        self.dual_memory.append(dual_info)
    
    def _update_adaptive_parameters(self, lp_success: bool, entity: Entity, 
                                  problem_state: ProblemState, constraints: List[Constraint]):
        """Update adaptive parameters based on LP success/failure and constraint satisfaction."""
        
        # Track constraint violations
        for constraint in constraints:
            attr = constraint.attribute
            current_count = problem_state.attribute_counts.get(attr, 0)
            if problem_state.accepted_count > 0:
                current_pct = current_count / problem_state.accepted_count
                violation = max(0, constraint.min_percentage - current_pct)
                self.constraint_violation_history[attr].append(violation)
        
        # Adapt acceptance threshold based on performance
        if lp_success:
            # LP succeeded - can be slightly more selective
            adjustment = -self.config.learning_rate * 0.5
        else:
            # LP failed - need to be less selective to find feasible solutions
            adjustment = self.config.learning_rate * 1.0
        
        # Consider constraint violation trends
        avg_violation = np.mean([
            np.mean(list(history)) for history in self.constraint_violation_history.values()
            if len(history) > 0
        ]) if self.constraint_violation_history else 0.0
        
        if avg_violation > 0.1:  # High violation - be more accepting
            adjustment += self.config.learning_rate * avg_violation
        
        # Update threshold with bounds
        self.current_acceptance_threshold += adjustment
        self.current_acceptance_threshold = np.clip(
            self.current_acceptance_threshold,
            self.config.min_acceptance_threshold,
            self.config.max_acceptance_threshold
        )
    
    def _reset_internal_state(self) -> None:
        """Reset adaptive state for new simulation."""
        self.dual_memory.clear()
        self.shadow_prices = {attr: 0.0 for attr in self.shadow_prices}
        self.constraint_tightness = {attr: 0.0 for attr in self.constraint_tightness}
        self.constraint_violation_history = {
            attr: deque(maxlen=20) for attr in self.constraint_violation_history
        }
        self.current_acceptance_threshold = self.config.acceptance_threshold
        self.infeasibility_count = 0
        self.total_decisions = 0
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return comprehensive strategy information including dual prices."""
        infeasibility_rate = self.infeasibility_count / max(1, self.total_decisions)
        
        return {
            "type": "AdaptiveLP",
            "acceptance_threshold": self.current_acceptance_threshold,
            "original_threshold": self.config.acceptance_threshold,
            "infeasibility_rate": infeasibility_rate,
            "shadow_prices": self.shadow_prices.copy(),
            "constraint_tightness": self.constraint_tightness.copy(),
            "dual_memory_size": len(self.dual_memory),
            "lookahead_horizon": self.config.lookahead_horizon,
            "adaptive_horizon": self.config.adaptive_horizon
        }


class RollingHorizonLPStrategy(AdaptiveLPStrategy):
    """
    Enhanced LP strategy with true rolling horizon and multi-stage decision making.
    
    This strategy maintains a rolling window of future decisions and updates them
    as new information becomes available.
    """
    
    def __init__(self, scenario: Scenario, config: Optional[AdaptiveLPConfig] = None):
        super().__init__(scenario, config)
        self._name = "RollingHorizonLP"
        
        # Rolling horizon state
        self.horizon_decisions: deque = deque(maxlen=self.config.lookahead_horizon)
        self.horizon_entities: deque = deque(maxlen=self.config.lookahead_horizon)
        self.receding_horizon_memory: List[Dict] = []
    
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: List[Constraint]) -> Decision:
        """Make decision using rolling horizon with multi-stage planning."""
        
        # Add current entity to horizon
        self.horizon_entities.append(entity)
        
        # Solve multi-stage problem over current horizon
        horizon_decisions = self._solve_rolling_horizon(problem_state, constraints)
        
        if horizon_decisions and len(horizon_decisions) > 0:
            # Take first decision from horizon solution
            decision = Decision.ACCEPT if horizon_decisions[0] >= self.current_acceptance_threshold else Decision.REJECT
            
            # Store remaining decisions for future use
            self.horizon_decisions = deque(horizon_decisions[1:], maxlen=self.config.lookahead_horizon)
            
            return decision
        else:
            # Fallback to adaptive LP single-stage decision
            return super()._decide_impl(entity, problem_state, constraints)
    
    def _solve_rolling_horizon(self, problem_state: ProblemState, constraints: List[Constraint]) -> List[float]:
        """Solve multi-stage LP over rolling horizon."""
        
        horizon_size = len(self.horizon_entities)
        if horizon_size == 0:
            return []
        
        try:
            # Multi-variable formulation: [x_1, x_2, ..., x_H] for each entity in horizon
            num_vars = horizon_size
            remaining_capacity = problem_state.capacity - problem_state.accepted_count
            
            # Objective: sum of rewards for each entity
            c = np.zeros(num_vars)
            for i, entity in enumerate(self.horizon_entities):
                c[i] = self._calculate_dual_aware_reward(entity, problem_state, constraints)
            
            # Constraints
            A_ub_list = []
            b_ub_list = []
            
            # Capacity constraint: sum(x_i) <= remaining_capacity
            capacity_constraint = np.ones(num_vars)
            A_ub_list.append(capacity_constraint)
            b_ub_list.append(remaining_capacity)
            
            # Attribute constraints for each constraint type
            for constraint in constraints:
                attr = constraint.attribute
                theta_j = constraint.min_percentage
                c_j_t = problem_state.attribute_counts.get(attr, 0)
                n_t = problem_state.accepted_count
                
                # Constraint: c_j_t + sum(x_i * has_attr_i) >= theta_j * (n_t + sum(x_i))
                # Rearranged: sum(x_i * (has_attr_i - theta_j)) >= theta_j * n_t - c_j_t
                
                attr_constraint = np.zeros(num_vars)
                for i, entity in enumerate(self.horizon_entities):
                    has_attr = 1.0 if entity.attributes.get(attr, False) else 0.0
                    attr_constraint[i] = -(has_attr - theta_j)  # Negate for <= form
                
                rhs = theta_j * n_t - c_j_t
                bound_value = -rhs + self.config.constraint_buffer
                
                A_ub_list.append(attr_constraint)
                b_ub_list.append(bound_value)
            
            # Solve multi-stage LP
            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
            bounds = [(0, 1)] * num_vars
            
            result = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.config.solver_method)
            
            if result.success:
                # Store solution information
                solution_info = {
                    'decisions': result.x.tolist(),
                    'objective': -result.fun,
                    'horizon_size': horizon_size,
                    'feasible': True
                }
                self.receding_horizon_memory.append(solution_info)
                
                return result.x.tolist()
            else:
                # Multi-stage LP failed - use single stage as fallback
                return []
                
        except Exception:
            return []
    
    def _reset_internal_state(self) -> None:
        """Reset rolling horizon state."""
        super()._reset_internal_state()
        self.horizon_decisions.clear()
        self.horizon_entities.clear()
        self.receding_horizon_memory.clear()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return rolling horizon strategy information."""
        info = super().get_strategy_info()
        info.update({
            "type": "RollingHorizonLP",
            "horizon_decisions_size": len(self.horizon_decisions),
            "horizon_entities_size": len(self.horizon_entities),
            "multi_stage_solutions": len(self.receding_horizon_memory)
        })
        return info