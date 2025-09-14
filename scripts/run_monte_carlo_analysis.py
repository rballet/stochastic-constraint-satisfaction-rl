#!/usr/bin/env python3
"""
Monte Carlo analysis of strategies for Stochastic Constraint Satisfaction Problems.

This script provides a clean, extensible framework for comparing strategy performance
using Monte Carlo simulation with statistical rigor.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
from enum import Enum

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.performance_analyzer import PerformanceAnalyzer, PerformanceReporter
from src.strategies.lp_strategy import LinearProgrammingStrategy, LPStrategyConfig
from src.strategies.adaptive_lp_strategy import AdaptiveLPStrategy, AdaptiveLPConfig
from src.strategies.advanced_lp_strategies import (
    RobustOptimizationStrategy, RobustOptimizationConfig
)
from src.examples.hospital_icu.scenarios import create_icu_scenario_1, create_icu_scenario_2, create_icu_scenario_3, create_icu_scenario_4
from src.core.strategy_base import AbstractStrategy, RandomStrategy
from src.strategies.greedy_strategies import (
    GreedyStrategy, WeightedConstraintGreedy, AdaptiveThresholdGreedy,
    WeightedGreedyConfig, AdaptiveGreedyConfig
)
from src.strategies.dqn_strategy import DQNStrategy, DQNConfig, create_dqn_strategy


class AnalysisType(Enum):
    """Types of analysis to run."""
    QUICK = "quick"
    FULL = "full"
    COMPREHENSIVE = "comprehensive"


@dataclass
class StrategyConfig:
    """Configuration for creating a strategy instance."""
    strategy_class: type
    name: str
    config_class: Optional[type] = None
    config_params: Optional[Dict[str, Any]] = None
    requires_scenario: bool = True


class StrategyFactory:
    """Factory for creating strategy instances with consistent configurations."""
    
    # Define strategy configurations once
    STRATEGY_DEFINITIONS = {
        # Greedy Strategies
        "greedy": StrategyConfig(
            strategy_class=GreedyStrategy,
            name="Greedy",
            requires_scenario=False
        ),
        "weighted_deficit": StrategyConfig(
            strategy_class=WeightedConstraintGreedy,
            name="WeightedGreedy-Def",
            config_class=WeightedGreedyConfig,
            config_params={"acceptance_threshold": 0.0, "weight_strategy": "deficit_proportional", "capacity_buffer": 0.0}
        ),
        "weighted_difficulty": StrategyConfig(
            strategy_class=WeightedConstraintGreedy,
            name="WeightedGreedy-Diff",
            config_class=WeightedGreedyConfig,
            config_params={"acceptance_threshold": 0.0, "weight_strategy": "constraint_difficulty", "capacity_buffer": 0.0}
        ),
        "adaptive_greedy": StrategyConfig(
            strategy_class=AdaptiveThresholdGreedy,
            name="AdaptiveGreedy",
            config_class=AdaptiveGreedyConfig,
            config_params={"base_threshold": 0.0, "capacity_pressure_factor": 0.3, "time_pressure_factor": 0.1, "capacity_buffer": 0.01},
            requires_scenario=False
        ),
        
        # Linear Programming Strategies
        "lp_conservative": StrategyConfig(
            strategy_class=LinearProgrammingStrategy,
            name="LP-Conservative",
            config_class=LPStrategyConfig,
            config_params={"acceptance_threshold": 0.8, "constraint_buffer": 0.1, "lookahead_horizon": 20}
        ),
        "lp_balanced": StrategyConfig(
            strategy_class=LinearProgrammingStrategy,
            name="LP-Balanced",
            config_class=LPStrategyConfig,
            config_params={"acceptance_threshold": 0.5, "constraint_buffer": 0.05, "lookahead_horizon": 15}
        ),
        
        # Advanced LP Strategies
        "adaptive_lp": StrategyConfig(
            strategy_class=AdaptiveLPStrategy,
            name="AdaptiveLP",
            config_class=AdaptiveLPConfig,
            config_params={"acceptance_threshold": 0.5, "dual_memory_length": 50, "learning_rate": 0.1, "adaptive_horizon": True}
        ),
        "robust_lp": StrategyConfig(
            strategy_class=RobustOptimizationStrategy,
            name="RobustLP",
            config_class=RobustOptimizationConfig,
            config_params={"acceptance_threshold": 0.6, "uncertainty_budget": 0.1, "worst_case_scenarios": 3}
        ),
        
        # Reinforcement Learning Strategies
        "dqn_pretrained": StrategyConfig(
            strategy_class=DQNStrategy,
            name="DQN-ICU",
            config_class=DQNConfig,
            config_params={"model_path": "models/icu_dqn/dqn_standard_5000000.zip"},
            requires_scenario=True
        ),
        
        # Random baseline
        "random": StrategyConfig(
            strategy_class=RandomStrategy,
            name="Random-70%",
            config_params={"acceptance_rate": 0.7},
            requires_scenario=False
        )
    }
    
    @classmethod
    def create_strategy(cls, strategy_key: str, scenario=None) -> AbstractStrategy:
        """Create a strategy instance from its configuration."""
        if strategy_key not in cls.STRATEGY_DEFINITIONS:
            raise ValueError(f"Unknown strategy: {strategy_key}")
        
        config = cls.STRATEGY_DEFINITIONS[strategy_key]
        
        # Create configuration object if needed
        strategy_config = None
        if config.config_class and config.config_params:
            strategy_config = config.config_class(**config.config_params)
        
        # Create strategy instance
        if config.requires_scenario and scenario is not None:
            if strategy_config:
                strategy = config.strategy_class(scenario, strategy_config)
            else:
                strategy = config.strategy_class(scenario)
        else:
            if strategy_config:
                strategy = config.strategy_class(strategy_config)
            elif config.config_params:
                strategy = config.strategy_class(**config.config_params)
            else:
                strategy = config.strategy_class()
        
        # Set the display name
        strategy._name = config.name
        return strategy
    
    @classmethod
    def get_strategy_sets(cls) -> Dict[AnalysisType, List[str]]:
        """Get predefined strategy sets for different analysis types."""
        return {
            AnalysisType.QUICK: [
                "greedy",
                "weighted_deficit", 
                "lp_balanced",
                "adaptive_lp",
                "dqn_pretrained"
            ],
            AnalysisType.FULL: [
                "greedy",
                "weighted_deficit",
                "weighted_difficulty", 
                "adaptive_greedy",
                "lp_conservative",
                "adaptive_lp",
                "robust_lp"
            ],
            AnalysisType.COMPREHENSIVE: [
                "random",
                "greedy",
                "weighted_deficit",
                "weighted_difficulty",
                "adaptive_greedy", 
                "lp_balanced",
                "lp_conservative",
                "adaptive_lp",
                "robust_lp",
                "dqn_pretrained"
            ]
        }


class ScenarioManager:
    """Manages scenario creation and metadata."""
    
    SCENARIOS = {
        "standard": ("Standard", create_icu_scenario_1),
        "high_acuity": ("High-Acuity", create_icu_scenario_2), 
        "emergency": ("Emergency", create_icu_scenario_3),
        "negative_correlations": ("Negative-Correlations", create_icu_scenario_4)
    }
    
    @classmethod
    def get_scenario(cls, scenario_key: str):
        """Get a scenario by key."""
        if scenario_key not in cls.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key}")
        
        name, factory = cls.SCENARIOS[scenario_key]
        return name, factory()
    
    @classmethod
    def get_all_scenarios(cls):
        """Get all available scenarios."""
        return [(name, factory()) for name, factory in cls.SCENARIOS.values()]


def create_strategies_for_scenario(scenario, analysis_type: AnalysisType = AnalysisType.FULL) -> List[AbstractStrategy]:
    """Create strategies for analysis using the factory pattern."""
    factory = StrategyFactory()
    strategy_keys = factory.get_strategy_sets()[analysis_type]
    
    return [factory.create_strategy(key, scenario) for key in strategy_keys]


class MonteCarloAnalyzer:
    """Main analyzer class following Single Responsibility Principle."""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.reporter = PerformanceReporter()
        
    def run_analysis(self, 
                    analysis_type: AnalysisType = AnalysisType.FULL,
                    num_runs: int = 100,
                    scenarios: Optional[List[str]] = None,
                    save_results: bool = False) -> List:
        """
        Run Monte Carlo analysis with specified parameters.
        
        Args:
            analysis_type: Type of analysis (quick, full, comprehensive)
            num_runs: Number of simulation runs per strategy-scenario combination
            scenarios: List of scenario keys to analyze (None = all scenarios)
            save_results: Whether to save detailed results
            
        Returns:
            List of performance metrics
        """
        print(f"Starting Monte Carlo Analysis ({analysis_type.value.upper()})")
        print(f"* Configuration: {num_runs} runs per strategy-scenario combination")
        print("="*80)
        
        # Get scenarios to analyze
        if scenarios:
            scenario_list = [(ScenarioManager.get_scenario(key)) for key in scenarios]
        else:
            scenario_list = ScenarioManager.get_all_scenarios()
        
        all_metrics = []
        
        # Run analysis for each scenario
        for scenario_name, scenario in scenario_list:
            print(f"\n🏥 Analyzing Scenario: {scenario_name}")
            print("-" * 50)
            
            strategies = create_strategies_for_scenario(scenario, analysis_type)
            
            for strategy in strategies:
                print(f"  🔄 Running {strategy.name}...")
                
                # Generate consistent seeds for fair comparison
                np.random.seed(42)
                seeds = np.random.randint(0, 1000000, num_runs).tolist()
                
                metrics = self.analyzer.analyze_strategy(
                    strategy=strategy,
                    scenario=scenario,
                    num_runs=num_runs,
                    seeds=seeds,
                    entity_generator_type="multivariate"
                )
                
                all_metrics.append(metrics)
                
                # Quick progress report
                print(f"    ✅ Success Rate: {metrics.success_rate:.1%}, "
                      f"Avg Accepted: {metrics.avg_accepted:.0f}, "
                      f"Runtime: {metrics.avg_runtime_seconds*1000:.1f}ms")
        
        print("\n" + "="*80)
        print("* GENERATING REPORT")
        print("="*80)
        
        # Generate comprehensive report
        self.reporter.print_comprehensive_report(all_metrics)
        
        # Save detailed results if requested
        if save_results:
            self._save_detailed_results(all_metrics)
        
        return all_metrics
    
    def _save_detailed_results(self, metrics_list):
        """Save detailed results to files."""
        print(f"\nResults would be saved here (CSV export disabled for now)")


def main():
    """Main entry point with improved CLI interface."""
    
    parser = argparse.ArgumentParser(
        description="Monte Carlo Strategy Analysis for Stochastic Constraint Satisfaction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Types:
  quick         - Fast analysis with 4 best strategies on 1 scenario
  full          - Complete analysis with 7 top strategies on all scenarios  
  comprehensive - Exhaustive analysis with all 9 strategies including baselines

Examples:
  python run_monte_carlo_analysis.py --type quick --runs 50
  python run_monte_carlo_analysis.py --type full --scenarios standard emergency
  python run_monte_carlo_analysis.py --type comprehensive --save
        """
    )
    
    parser.add_argument("--type", 
                       choices=["quick", "full", "comprehensive"],
                       default="full",
                       help="Type of analysis to run (default: full)")
    parser.add_argument("--runs", 
                       type=int, 
                       default=100, 
                       help="Number of simulation runs per strategy (default: 100)")
    parser.add_argument("--scenarios",
                       nargs="+",
                       choices=["standard", "high_acuity", "emergency", "negative_correlations"],
                       help="Specific scenarios to analyze (default: all)")
    parser.add_argument("--save", 
                       action="store_true", 
                       help="Save detailed results to files")
    
    args = parser.parse_args()
    
    try:
        # Create analyzer with specified capacity
        analyzer = MonteCarloAnalyzer()
        
        # Map string to enum
        analysis_type = AnalysisType(args.type)
        
        # Run analysis
        analyzer.run_analysis(
            analysis_type=analysis_type,
            num_runs=args.runs,
            scenarios=args.scenarios,
            save_results=args.save
        )
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
