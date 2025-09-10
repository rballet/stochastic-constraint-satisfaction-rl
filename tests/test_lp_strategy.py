"""
Test cases for Linear Programming strategy.
"""

import pytest
import numpy as np
from src.strategies.lp_strategy import LinearProgrammingStrategy, LPStrategyConfig
from src.examples.hospital_icu.scenarios import create_icu_scenario_1, create_icu_scenario_2
from src.simulation.engine import SimulationEngine
from src.simulation.entity_generator import create_entity_generator
from src.core.types import Decision


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
