#!/usr/bin/env python3
"""
Monte Carlo analysis of top-performing strategies on ICU scenarios.

This script runs performance analysis comparing the best-performing
strategies using Monte Carlo simulation with statistical rigor.
Only well-performing strategies are included for cleaner analysis.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.performance_analyzer import PerformanceAnalyzer, PerformanceReporter
from src.strategies.lp_strategy import LinearProgrammingStrategy, LPStrategyConfig
from src.strategies.adaptive_lp_strategy import AdaptiveLPStrategy, RollingHorizonLPStrategy, AdaptiveLPConfig
from src.strategies.advanced_lp_strategies import (
    MultiStageStochasticStrategy, RobustOptimizationStrategy, ChanceConstraintStrategy,
    MultiStageConfig, RobustOptimizationConfig, ChanceConstraintConfig
)
from src.examples.hospital_icu.scenarios import create_icu_scenario_1, create_icu_scenario_2, create_icu_scenario_3
from src.core.strategy_base import AbstractStrategy
from src.core.types import Entity, ProblemState, Constraint, Decision
from src.strategies.greedy_strategies import GreedyStrategy
from src.core.strategy_base import RandomStrategy


def create_strategies_for_scenario(scenario):
    """Create well-performing strategies for comparison."""
    
    strategies = [
        # Best performing heuristic strategy
        GreedyStrategy(),
        
        # Best performing LP strategy (conservative configuration)
        LinearProgrammingStrategy(scenario, LPStrategyConfig(
            acceptance_threshold=0.8,
            constraint_buffer=0.1,
            lookahead_horizon=20
        )),
        
        # Best advanced LP strategies
        AdaptiveLPStrategy(scenario, AdaptiveLPConfig(
            acceptance_threshold=0.5,
            dual_memory_length=50,
            learning_rate=0.1,
            adaptive_horizon=True
        )),
        
        RobustOptimizationStrategy(scenario, RobustOptimizationConfig(
            acceptance_threshold=0.6,
            uncertainty_budget=0.1,
            worst_case_scenarios=3
        )),
    ]
    
    # Update strategy names
    for strategy in strategies:
        if isinstance(strategy, LinearProgrammingStrategy):
            strategy._name = "LP-Conservative"
        elif isinstance(strategy, AdaptiveLPStrategy):
            strategy._name = "AdaptiveLP"
        elif isinstance(strategy, RobustOptimizationStrategy):
            strategy._name = "RobustLP"
        elif isinstance(strategy, GreedyStrategy):
            strategy._name = "Greedy"
    
    return strategies


def run_full_analysis(num_runs: int = 100, save_results: bool = True):
    """Run Monte Carlo analysis."""
    
    print("Starting Monte Carlo Analysis")
    print(f"* Configuration: {num_runs} runs per strategy-scenario combination")
    print("="*80)
    
    # Define scenarios
    scenarios = [
        ("Standard", create_icu_scenario_1()),
        ("High-Acuity", create_icu_scenario_2()),
        ("Emergency", create_icu_scenario_3())
    ]
    
    # Initialize analyzer and reporter
    analyzer = PerformanceAnalyzer(capacity=1000, max_rejections=2000)
    reporter = PerformanceReporter()
    
    all_metrics = []
    
    # Run analysis for each scenario
    for scenario_name, scenario in scenarios:
        print(f"\n🏥 Analyzing Scenario: {scenario_name}")
        print("-" * 50)
        
        strategies = create_strategies_for_scenario(scenario)
        
        for strategy in strategies:
            print(f"  🔄 Running {strategy.name}...")
            
            # Generate consistent seeds for fair comparison
            np.random.seed(42)
            seeds = np.random.randint(0, 1000000, num_runs).tolist()
            
            metrics = analyzer.analyze_strategy(
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
    print("📈 GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    # Generate comprehensive report
    reporter.print_comprehensive_report(all_metrics)
    
    # Save detailed results if requested
    if save_results:
        save_detailed_results(all_metrics, reporter)
    
    return all_metrics


def save_detailed_results(metrics_list, reporter):
    """Save detailed results to files."""
    
    print(f"\n💾 Results would be saved here (CSV export disabled for now)")
    print("   - Enable pandas dependency to save CSV files")


def run_quick_analysis(num_runs: int = 20):
    """Run quick analysis for testing."""
    
    print("⚡ Running Quick Analysis")
    print("="*50)
    
    scenario = create_icu_scenario_1()
    strategies = [
        RandomStrategy(0.7),
        GreedyStrategy(),
        LinearProgrammingStrategy(scenario, LPStrategyConfig(acceptance_threshold=0.5)),
        AdaptiveLPStrategy(scenario, AdaptiveLPConfig(
            acceptance_threshold=0.5,
            learning_rate=0.1,
            adaptive_horizon=True
        )),
        MultiStageStochasticStrategy(scenario, MultiStageConfig(
            acceptance_threshold=0.5,
            scenario_tree_size=3
        )),
        ChanceConstraintStrategy(scenario, ChanceConstraintConfig(
            acceptance_threshold=0.5,
            confidence_level=0.85
        ))
    ]
    strategies[2]._name = "LP-Balanced"
    strategies[3]._name = "AdaptiveLP"
    strategies[4]._name = "MultiStageLP"
    strategies[5]._name = "ChanceConstraintLP"
    
    analyzer = PerformanceAnalyzer()
    reporter = PerformanceReporter()
    
    metrics_list = []
    
    for strategy in strategies:
        print(f"🔄 Analyzing {strategy.name}...")
        
        metrics = analyzer.analyze_strategy(
            strategy=strategy,
            scenario=scenario,
            num_runs=num_runs,
            entity_generator_type="multivariate"
        )
        
        metrics_list.append(metrics)
    
    reporter.print_comprehensive_report(metrics_list)
    
    return metrics_list


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Monte Carlo Strategy Analysis")
    parser.add_argument("--runs", type=int, default=100, help="Number of simulation runs per strategy")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis (20 runs, single scenario)")
    parser.add_argument("--save", action="store_true", help="Save results to CSV files")
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            run_quick_analysis(20)
        else:
            run_full_analysis(args.runs, args.save)
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
