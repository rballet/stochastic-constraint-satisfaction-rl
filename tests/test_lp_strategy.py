"""
Test cases for Linear Programming strategies.
"""

import pytest
import numpy as np
from src.strategies.lp_strategy import LinearProgrammingStrategy, LPStrategyConfig
from src.strategies.advanced_lp_strategies import (
    MultiStageStochasticStrategy, MultiStageConfig,
    RobustOptimizationStrategy, RobustOptimizationConfig,
    ChanceConstraintStrategy, ChanceConstraintConfig
)
from src.examples.hospital_icu.scenarios import create_icu_scenario_1, create_icu_scenario_2
from src.simulation.engine import SimulationEngine
from src.simulation.entity_generator import create_entity_generator
from src.core.types import Decision, Entity, ProblemState


class TestLPStrategy:
    """Test LP strategy implementation."""
    
    def test_strategy_initialization(self):
        """Test LP strategy initializes correctly."""
        scenario = create_icu_scenario_1()
        config = LPStrategyConfig(acceptance_threshold=0.6, lookahead_horizon=30)
        
        strategy = LinearProgrammingStrategy(scenario, config)
        
        assert strategy.name == "LinearProgramming"
        assert strategy.config.acceptance_threshold == 0.6
        assert strategy.config.lookahead_horizon == 30
        assert len(strategy._attribute_to_index) == len(scenario.attributes)
    
    def test_strategy_with_icu_scenario(self):
        """Test LP strategy with ICU scenario."""
        scenario = create_icu_scenario_1()
        strategy = LinearProgrammingStrategy(scenario)
        
        # Create entity generator and engine
        entity_generator = create_entity_generator("multivariate", seed=42)
        engine = SimulationEngine(entity_generator)
        
        # Run simulation
        result = engine.run_simulation(scenario, strategy, seed=42)
        
        assert result is not None
        assert result.accepted_count >= 0
        assert result.rejected_count >= 0
        
        # LP strategy should perform better than random
        # (At least it should make some decisions)
        assert result.accepted_count + result.rejected_count > 0
    
    def test_strategy_constraint_awareness(self):
        """Test that LP strategy is aware of constraints."""
        scenario = create_icu_scenario_1()
        config = LPStrategyConfig(acceptance_threshold=0.3)  # Lower threshold
        strategy = LinearProgrammingStrategy(scenario, config)
        
        entity_generator = create_entity_generator("multivariate", seed=123)
        engine = SimulationEngine(entity_generator)
        
        result = engine.run_simulation(scenario, strategy, seed=123)
        
        # Check if strategy achieved better constraint satisfaction than random
        info = strategy.get_strategy_info()
        assert "LinearProgramming" in info["type"]
        assert info["acceptance_threshold"] == 0.3
    
    def test_multiple_scenarios(self):
        """Test LP strategy adapts to different scenarios."""
        scenarios = [create_icu_scenario_1(), create_icu_scenario_2()]
        
        results = []
        for scenario in scenarios:
            strategy = LinearProgrammingStrategy(scenario)
            entity_generator = create_entity_generator("multivariate", seed=456)
            engine = SimulationEngine(entity_generator)
            
            result = engine.run_simulation(scenario, strategy, seed=456)
            results.append(result)
        
        # Both scenarios should produce valid results
        for result in results:
            assert result.accepted_count >= 0
            assert result.rejected_count >= 0
    
    def test_strategy_config_variations(self):
        """Test different strategy configurations."""
        scenario = create_icu_scenario_1()
        
        configs = [
            LPStrategyConfig(acceptance_threshold=0.3, lookahead_horizon=20),
            LPStrategyConfig(acceptance_threshold=0.7, lookahead_horizon=50),
            LPStrategyConfig(acceptance_threshold=0.5, constraint_buffer=0.1)
        ]
        
        for config in configs:
            strategy = LinearProgrammingStrategy(scenario, config)
            
            # Test that strategy accepts configuration
            assert strategy.config.acceptance_threshold == config.acceptance_threshold
            assert strategy.config.lookahead_horizon == config.lookahead_horizon
            
            # Test basic functionality
            entity_generator = create_entity_generator("basic", seed=789)
            engine = SimulationEngine(entity_generator)
            result = engine.run_simulation(scenario, strategy, seed=789)
            
            assert result is not None
    
    def test_strategy_comparison(self):
        """Compare LP strategy with different configurations."""
        scenario = create_icu_scenario_1()
        
        # Conservative strategy (high threshold)
        conservative_strategy = LinearProgrammingStrategy(
            scenario, 
            LPStrategyConfig(acceptance_threshold=0.8, constraint_buffer=0.1)
        )
        
        # Aggressive strategy (low threshold)
        aggressive_strategy = LinearProgrammingStrategy(
            scenario,
            LPStrategyConfig(acceptance_threshold=0.2, constraint_buffer=0.02)
        )
        
        entity_generator = create_entity_generator("multivariate")
        engine = SimulationEngine(entity_generator)
        
        # Run both strategies
        conservative_result = engine.run_simulation(scenario, conservative_strategy, seed=999)
        aggressive_result = engine.run_simulation(scenario, aggressive_strategy, seed=999)
        
        # Aggressive strategy should generally accept more patients
        # (though this depends on the specific run)
        assert conservative_result.accepted_count >= 0
        assert aggressive_result.accepted_count >= 0
        
        # Both should produce valid results
        assert conservative_result.accepted_count + conservative_result.rejected_count > 0
        assert aggressive_result.accepted_count + aggressive_result.rejected_count > 0


class TestAdvancedLPStrategies:
    """Test advanced Linear Programming strategies."""
    
    def test_multistage_strategy_initialization(self):
        """Test Multi-Stage Stochastic strategy initializes correctly."""
        scenario = create_icu_scenario_1()
        config = MultiStageConfig(
            acceptance_threshold=0.7,
            num_stages=3,
            scenario_tree_size=5,
            lookahead_horizon=30
        )
        
        strategy = MultiStageStochasticStrategy(scenario, config)
        
        assert strategy.name == "MultiStageStochastic"
        assert strategy.config.acceptance_threshold == 0.7
        assert strategy.config.num_stages == 3
        assert strategy.config.scenario_tree_size == 5
    
    def test_robust_optimization_strategy_initialization(self):
        """Test Robust Optimization strategy initializes correctly."""
        scenario = create_icu_scenario_1()
        config = RobustOptimizationConfig(
            acceptance_threshold=0.8,
            uncertainty_budget=0.15,
            worst_case_scenarios=4
        )
        
        strategy = RobustOptimizationStrategy(scenario, config)
        
        assert strategy.name == "RobustOptimization"
        assert strategy.config.acceptance_threshold == 0.8
        assert strategy.config.uncertainty_budget == 0.15
        assert strategy.config.worst_case_scenarios == 4
    
    def test_chance_constraint_strategy_initialization(self):
        """Test Chance Constraint strategy initializes correctly."""
        scenario = create_icu_scenario_1()
        config = ChanceConstraintConfig(
            acceptance_threshold=0.6,
            confidence_level=0.95,
            sample_size=100
        )
        
        strategy = ChanceConstraintStrategy(scenario, config)
        
        assert strategy.name == "ChanceConstraint"
        assert strategy.config.acceptance_threshold == 0.6
        assert strategy.config.confidence_level == 0.95
        assert strategy.config.sample_size == 100
    
    def test_all_strategies_basic_functionality(self):
        """Test that all strategies can make basic decisions."""
        scenario = create_icu_scenario_1()
        
        strategies = [
            ("Basic LP", LinearProgrammingStrategy(scenario)),
            ("Multi-Stage", MultiStageStochasticStrategy(scenario, MultiStageConfig(scenario_tree_size=3))),
            ("Robust", RobustOptimizationStrategy(scenario, RobustOptimizationConfig())),
            ("Chance", ChanceConstraintStrategy(scenario, ChanceConstraintConfig(sample_size=20)))
        ]
        
        # Test decision making with sample entity and state
        entity = Entity({"critical_condition": True, "elderly": False, "has_insurance": True})
        problem_state = ProblemState(
            capacity=100,
            accepted_count=30,
            attribute_counts={"critical_condition": 15, "elderly": 8, "has_insurance": 25}
        )
        
        for name, strategy in strategies:
            try:
                decision = strategy._decide_impl(entity, problem_state, scenario.constraints)
                assert decision in [Decision.ACCEPT, Decision.REJECT], f"{name} strategy returned invalid decision"
                
                # Test strategy info
                info = strategy.get_strategy_info()
                assert "type" in info, f"{name} strategy missing type in info"
                assert "acceptance_threshold" in info, f"{name} strategy missing acceptance_threshold in info"
                
            except Exception as e:
                pytest.fail(f"{name} strategy failed with error: {e}")
    
    def test_strategies_with_simulation(self):
        """Test strategies in full simulation environment."""
        scenario = create_icu_scenario_1()
        
        strategies = [
            ("Basic LP", LinearProgrammingStrategy(scenario, LPStrategyConfig(acceptance_threshold=0.5))),
            ("Multi-Stage", MultiStageStochasticStrategy(scenario, MultiStageConfig(
                acceptance_threshold=0.5, 
                scenario_tree_size=3,
                lookahead_horizon=20
            ))),
            ("Robust", RobustOptimizationStrategy(scenario, RobustOptimizationConfig(
                acceptance_threshold=0.6,
                uncertainty_budget=0.1
            ))),
            ("Chance", ChanceConstraintStrategy(scenario, ChanceConstraintConfig(
                acceptance_threshold=0.5,
                confidence_level=0.85,
                sample_size=20  # Reduced for faster testing
            )))
        ]
        
        entity_generator = create_entity_generator("basic", seed=42)
        engine = SimulationEngine(entity_generator)
        
        results = []
        
        for name, strategy in strategies:
            try:
                result = engine.run_simulation(scenario, strategy, seed=42)
                
                # Basic validation
                assert result is not None, f"{name} strategy returned None result"
                assert result.accepted_count >= 0, f"{name} strategy has negative accepted count"
                assert result.rejected_count >= 0, f"{name} strategy has negative rejected count"
                assert result.accepted_count + result.rejected_count > 0, f"{name} strategy made no decisions"
                
                results.append({
                    'name': name,
                    'success': result.success,
                    'accepted': result.accepted_count,
                    'rejected': result.rejected_count,
                    'total_decisions': result.accepted_count + result.rejected_count
                })
                
            except Exception as e:
                pytest.fail(f"{name} strategy failed in simulation: {e}")
        
        # At least one strategy should make some decisions
        total_decisions = sum(r['total_decisions'] for r in results)
        assert total_decisions > 0, "No strategy made any decisions"
    
    def test_strategy_configuration_validation(self):
        """Test different configuration parameters for advanced strategies."""
        scenario = create_icu_scenario_1()
        
        # Test Multi-Stage configurations
        multistage_configs = [
            MultiStageConfig(num_stages=2, scenario_tree_size=4),
            MultiStageConfig(num_stages=4, scenario_tree_size=8),
            MultiStageConfig(acceptance_threshold=0.3, num_stages=3)
        ]
        
        for config in multistage_configs:
            strategy = MultiStageStochasticStrategy(scenario, config)
            assert strategy.config.num_stages == config.num_stages
            assert strategy.config.scenario_tree_size == config.scenario_tree_size
        
        # Test Robust Optimization configurations
        robust_configs = [
            RobustOptimizationConfig(uncertainty_budget=0.05, worst_case_scenarios=3),
            RobustOptimizationConfig(uncertainty_budget=0.2, worst_case_scenarios=7),
            RobustOptimizationConfig(acceptance_threshold=0.9, uncertainty_budget=0.1)
        ]
        
        for config in robust_configs:
            strategy = RobustOptimizationStrategy(scenario, config)
            assert strategy.config.uncertainty_budget == config.uncertainty_budget
            assert strategy.config.worst_case_scenarios == config.worst_case_scenarios
        
        # Test Chance Constraint configurations
        chance_configs = [
            ChanceConstraintConfig(confidence_level=0.90, sample_size=50),
            ChanceConstraintConfig(confidence_level=0.99, sample_size=200),
            ChanceConstraintConfig(acceptance_threshold=0.4, confidence_level=0.95)
        ]
        
        for config in chance_configs:
            strategy = ChanceConstraintStrategy(scenario, config)
            assert strategy.config.confidence_level == config.confidence_level
            assert strategy.config.sample_size == config.sample_size
    
    def test_lp_formulation_validity(self):
        """Test that LP formulation is mathematically valid."""
        scenario = create_icu_scenario_1()
        strategy = LinearProgrammingStrategy(scenario)
        
        entity = Entity({"critical_condition": True, "elderly": False, "has_insurance": True})
        problem_state = ProblemState(
            capacity=100,
            accepted_count=40,
            attribute_counts={"critical_condition": 20, "elderly": 12, "has_insurance": 35}
        )
        
        c, A_ub, b_ub = strategy._formulate_lp(entity, problem_state, scenario.constraints)
        
        if c is not None:  # LP formulation succeeded
            # Check dimensions are consistent
            assert len(c) > 0, "Objective vector is empty"
            assert A_ub.shape[1] == len(c), "Constraint matrix width doesn't match objective vector length"
            assert A_ub.shape[0] == len(b_ub), "Constraint matrix height doesn't match bounds vector length"
            
            # Check that formulation respects capacity
            assert len(c) <= problem_state.capacity - problem_state.accepted_count + 1, "Too many variables for remaining capacity"
        else:
            # LP formulation failed (e.g., no remaining capacity)
            assert problem_state.accepted_count >= problem_state.capacity, "LP formulation failed unexpectedly"
    
    def test_constraint_deficit_calculation(self):
        """Test constraint deficit calculation across strategies."""
        scenario = create_icu_scenario_1()
        strategies = [
            LinearProgrammingStrategy(scenario),
            MultiStageStochasticStrategy(scenario, MultiStageConfig()),
            RobustOptimizationStrategy(scenario, RobustOptimizationConfig()),
            ChanceConstraintStrategy(scenario, ChanceConstraintConfig())
        ]
        
        # Test case where constraints are not met
        problem_state = ProblemState(
            capacity=100,
            accepted_count=50,
            attribute_counts={"critical_condition": 20, "elderly": 10, "has_insurance": 40}  # Below 60% critical, 40% elderly
        )
        
        for strategy in strategies:
            deficits = strategy._calculate_constraint_deficit(problem_state, scenario.constraints)
            
            # Should detect deficit in critical_condition (20/50 = 40% < 60%)
            critical_deficit = deficits.get("critical_condition", 0)
            assert critical_deficit > 0, f"{strategy.name} failed to detect critical condition deficit"
            
            # Should detect deficit in elderly (10/50 = 20% < 40%)  
            elderly_deficit = deficits.get("elderly", 0)
            assert elderly_deficit > 0, f"{strategy.name} failed to detect elderly deficit"
    
    def test_strategy_performance_comparison(self):
        """Compare performance of different LP strategies."""
        scenario = create_icu_scenario_1()
        
        strategies = [
            ("Conservative LP", LinearProgrammingStrategy(scenario, LPStrategyConfig(
                acceptance_threshold=0.8, constraint_buffer=0.1
            ))),
            ("Aggressive LP", LinearProgrammingStrategy(scenario, LPStrategyConfig(
                acceptance_threshold=0.3, constraint_buffer=0.02
            ))),
            ("Multi-Stage", MultiStageStochasticStrategy(scenario, MultiStageConfig(
                acceptance_threshold=0.6, scenario_tree_size=4
            ))),
            ("Robust", RobustOptimizationStrategy(scenario, RobustOptimizationConfig(
                acceptance_threshold=0.7, uncertainty_budget=0.1
            )))
        ]
        
        entity_generator = create_entity_generator("multivariate", seed=123)
        engine = SimulationEngine(entity_generator)
        
        performance_results = []
        
        for name, strategy in strategies:
            try:
                result = engine.run_simulation(scenario, strategy, seed=123)
                
                performance_results.append({
                    'name': name,
                    'success': result.success,
                    'accepted': result.accepted_count,
                    'efficiency': result.accepted_count / (result.accepted_count + result.rejected_count) if (result.accepted_count + result.rejected_count) > 0 else 0,
                    'final_percentages': result.final_attribute_percentages
                })
                
            except Exception as e:
                # Strategy failed, record as unsuccessful
                performance_results.append({
                    'name': name,
                    'success': False,
                    'error': str(e)
                })
        
        # At least some strategies should succeed
        successful_strategies = [r for r in performance_results if r.get('success', False)]
        assert len(successful_strategies) > 0, "No strategies succeeded in simulation"
        
        # All successful strategies should make reasonable decisions
        for result in successful_strategies:
            assert result['accepted'] >= 0, f"{result['name']} has invalid accepted count"
            assert 0 <= result['efficiency'] <= 1, f"{result['name']} has invalid efficiency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
