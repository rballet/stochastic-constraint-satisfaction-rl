#!/usr/bin/env python3
"""
Monte Carlo analysis of strategies on ICU scenarios.

This script runs comprehensive performance analysis comparing different
strategies using Monte Carlo simulation with statistical rigor.
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
from src.examples.hospital_icu.scenarios import create_icu_scenario_1, create_icu_scenario_2, create_icu_scenario_3
from src.core.strategy_base import AbstractStrategy
from src.core.types import Entity, ProblemState, Constraint, Decision


class RandomStrategy(AbstractStrategy):
    """Random strategy for baseline comparison."""
    
    def __init__(self, acceptance_rate: float = 0.7, name_suffix: str = ""):
        name = f"Random-{acceptance_rate:.0%}{name_suffix}"
        super().__init__(name)
        self.acceptance_rate = acceptance_rate
        self.random = np.random.RandomState(42)
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: list[Constraint]) -> Decision:
        return Decision.ACCEPT if self.random.random() < self.acceptance_rate else Decision.REJECT
    
    def _reset_internal_state(self) -> None:
        self.random = np.random.RandomState(42)


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


class PriorityStrategy(AbstractStrategy):
    """Priority-based strategy favoring critical attributes."""
    
    def __init__(self, priority_weights: dict = None):
        super().__init__("Priority")
        self.priority_weights = priority_weights or {
            'critical_condition': 3.0,
            'elderly': 2.0,
            'high_risk': 2.0,
            'emergency_case': 1.5,
            'has_insurance': 0.5
        }
        
    def _decide_impl(self, entity: Entity, problem_state: ProblemState, constraints: list[Constraint]) -> Decision:
        # Calculate priority score for entity
        priority_score = sum(
            self.priority_weights.get(attr, 1.0) * (1 if entity.attributes.get(attr, False) else 0)
            for attr in entity.attributes.keys()
        )
        
        # Accept if priority score is above threshold
        # Threshold depends on current constraint satisfaction
        base_threshold = 2.0
        
        # Lower threshold if constraints are not satisfied
        for constraint in constraints:
            if problem_state.accepted_count > 0:
                current_pct = problem_state.get_attribute_percentage(constraint.attribute)
                if current_pct < constraint.min_percentage:
                    # Need more of this attribute - lower threshold if entity has it
                    if entity.attributes.get(constraint.attribute, False):
                        base_threshold *= 0.7
        
        return Decision.ACCEPT if priority_score >= base_threshold else Decision.REJECT
    
    def _reset_internal_state(self) -> None:
        pass


def create_strategies_for_scenario(scenario):
    """Create all strategies for a given scenario."""
    
    strategies = [
        # Baseline strategies
        RandomStrategy(0.5),
        RandomStrategy(0.7),
        RandomStrategy(0.9),
        
        # Heuristic strategies
        GreedyStrategy(),
        PriorityStrategy(),
        
        # LP strategies with different configurations
        LinearProgrammingStrategy(scenario, LPStrategyConfig(
            acceptance_threshold=0.8,
            constraint_buffer=0.1,
            lookahead_horizon=20
        )),
        LinearProgrammingStrategy(scenario, LPStrategyConfig(
            acceptance_threshold=0.6,
            constraint_buffer=0.05,
            lookahead_horizon=30
        )),
        LinearProgrammingStrategy(scenario, LPStrategyConfig(
            acceptance_threshold=0.4,
            constraint_buffer=0.02,
            lookahead_horizon=50
        )),
        LinearProgrammingStrategy(scenario, LPStrategyConfig(
            acceptance_threshold=0.3,
            constraint_buffer=0.01,
            lookahead_horizon=100
        )),
    ]
    
    # Update strategy names for LP strategies
    lp_configs = ["Conservative", "Balanced", "Aggressive", "Very-Aggressive"]
    lp_strategies = [s for s in strategies if isinstance(s, LinearProgrammingStrategy)]
    for i, strategy in enumerate(lp_strategies):
        strategy._name = f"LP-{lp_configs[i]}"
    
    return strategies


def run_full_analysis(num_runs: int = 100, save_results: bool = True):
    """Run comprehensive Monte Carlo analysis."""
    
    print("🚀 Starting Comprehensive Monte Carlo Analysis")
    print(f"📊 Configuration: {num_runs} runs per strategy-scenario combination")
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
        LinearProgrammingStrategy(scenario, LPStrategyConfig(acceptance_threshold=0.5))
    ]
    strategies[-1]._name = "LP-Balanced"
    
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
