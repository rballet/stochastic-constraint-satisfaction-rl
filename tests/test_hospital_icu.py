"""
Test cases for hospital ICU admission system.
"""

import pytest
import numpy as np
from src.examples.hospital_icu.scenarios import create_icu_scenario_1
from src.examples.hospital_icu.config import create_config, EnvironmentConfig, RewardConfig
from src.examples.hospital_icu.icu_env import ICUAdmissionEnv
from src.core.strategy_base import AbstractStrategy
from src.core.types import Entity, ProblemState, Constraint, Decision
from src.simulation.engine import SimulationEngine
from src.simulation.entity_generator import create_entity_generator


class DummyStrategy(AbstractStrategy):
    """Simple test strategy."""
    
    def __init__(self, acceptance_rate: float = 0.7):
        super().__init__("DummyStrategy")
        self.acceptance_rate = acceptance_rate
        self.random = np.random.RandomState(42)
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: list[Constraint]) -> Decision:
        return Decision.ACCEPT if self.random.random() < self.acceptance_rate else Decision.REJECT
    
    def _reset_internal_state(self) -> None:
        self.random = np.random.RandomState(42)


class TestICUConfiguration:
    """Test ICU configuration system."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        env_config, reward_config = create_config()
        
        assert env_config.capacity == 50
        assert env_config.max_rejections == 200
        assert reward_config.constraint_penalty == 2.0
        assert reward_config.success_bonus == 10.0
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        env_config, reward_config = create_config(
            capacity=100,
            constraint_penalty=5.0,
            early_bonus=1.0
        )
        
        assert env_config.capacity == 100
        assert reward_config.constraint_penalty == 5.0
        assert reward_config.early_bonus == 1.0
    
    def test_config_immutability(self):
        """Test that configs are immutable."""
        env_config, _ = create_config()
        
        with pytest.raises(AttributeError):
            env_config.capacity = 200  # Should fail - frozen dataclass


class TestICUEnvironment:
    """Test ICU RL environment."""
    
    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        scenario = create_icu_scenario_1()
        env_config, reward_config = create_config(capacity=30, max_rejections=100)
        
        env = ICUAdmissionEnv(scenario, env_config, reward_config, seed=42)
        
        assert env.env_config.capacity == 30
        assert env.reward_config.constraint_penalty == 2.0
        assert env.action_space_size == 2
        assert len(scenario.constraints) == 3
    
    def test_environment_step(self):
        """Test environment step function."""
        scenario = create_icu_scenario_1()
        env_config, reward_config = create_config(capacity=10)
        
        env = ICUAdmissionEnv(scenario, env_config, reward_config, seed=42)
        obs = env.reset()
        
        assert obs is not None
        assert len(obs) == env.observation_space_size
        
        # Take a few steps
        for _ in range(5):
            action = 1  # Accept
            obs, reward, done, info = env.step(action)
            
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert 'accepted_count' in info
            assert 'rejected_count' in info
            
            if done:
                break
    
    def test_environment_with_default_config(self):
        """Test environment works with default config."""
        scenario = create_icu_scenario_1()
        env = ICUAdmissionEnv(scenario, seed=42)  # Should use defaults
        
        obs = env.reset()
        assert obs is not None
        assert env.env_config.capacity == 50  # Default


class TestICUSimulation:
    """Test ICU simulation engine integration."""
    
    def test_simulation_runs(self):
        """Test basic simulation execution."""
        scenario = create_icu_scenario_1()
        strategy = DummyStrategy(acceptance_rate=0.8)
        
        entity_generator = create_entity_generator("multivariate", seed=42)
        engine = SimulationEngine(entity_generator)
        
        result = engine.run_simulation(scenario, strategy, seed=42)
        
        assert result is not None
        assert result.accepted_count >= 0
        assert result.rejected_count >= 0
        assert isinstance(result.constraints_satisfied, bool)
    
    def test_multiple_simulations(self):
        """Test running multiple simulations."""
        scenario = create_icu_scenario_1()
        strategy = DummyStrategy()
        
        entity_generator = create_entity_generator("multivariate")
        engine = SimulationEngine(entity_generator)
        
        results = engine.run_multiple_simulations(
            scenario, strategy, num_runs=3, seeds=[42, 123, 456]
        )
        
        assert len(results) == 3
        for result in results:
            assert result.accepted_count >= 0
            assert result.rejected_count >= 0


class TestICUScenarios:
    """Test ICU scenario definitions."""
    
    def test_scenario_creation(self):
        """Test scenario is created correctly."""
        scenario = create_icu_scenario_1()
        
        assert scenario.name == "icu_standard"
        assert len(scenario.attributes) == 5
        assert len(scenario.constraints) == 3
        assert "critical_condition" in scenario.attributes
        assert "elderly" in scenario.attributes
    
    def test_scenario_constraints(self):
        """Test scenario constraints are properly defined."""
        scenario = create_icu_scenario_1()
        
        constraint_attrs = [c.attribute for c in scenario.constraints]
        assert "critical_condition" in constraint_attrs
        assert "elderly" in constraint_attrs
        assert "high_risk" in constraint_attrs
        
        # Check constraint thresholds
        critical_constraint = next(c for c in scenario.constraints if c.attribute == "critical_condition")
        assert critical_constraint.min_percentage == 0.60


if __name__ == "__main__":
    pytest.main([__file__])
