"""
Performance analysis tools for stochastic constraint satisfaction strategies.

Provides comprehensive metrics and statistical analysis for comparing
different strategies across multiple scenarios and runs.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import time

from ..core.types import SimulationResult, Scenario
from ..core.strategy_base import AbstractStrategy
from ..simulation.engine import SimulationEngine
from ..simulation.entity_generator import create_entity_generator


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a strategy."""
    
    # Basic metrics
    strategy_name: str
    scenario_name: str
    num_runs: int
    
    # Success metrics
    success_rate: float
    constraint_satisfaction_rate: float
    
    # Acceptance/Rejection metrics
    avg_accepted: float
    std_accepted: float
    avg_rejected: float
    std_rejected: float
    acceptance_rate: float  # accepted / (accepted + rejected)
    
    # Capacity utilization
    capacity_utilization: float  # avg_accepted / capacity
    capacity_utilization_std: float
    
    # Constraint satisfaction details
    constraint_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical metrics
    median_accepted: float = 0.0
    min_accepted: int = 0
    max_accepted: int = 0
    
    # Efficiency metrics
    avg_entities_processed: float = 0.0
    rejection_efficiency: float = 0.0  # How well does strategy use rejections
    
    # Runtime performance
    avg_runtime_seconds: float = 0.0
    std_runtime_seconds: float = 0.0
    
    # Confidence intervals
    acceptance_ci_95: Tuple[float, float] = (0.0, 0.0)
    success_rate_ci_95: Tuple[float, float] = (0.0, 0.0)


class PerformanceAnalyzer:
    """Comprehensive performance analysis for strategies."""
    
    def __init__(self, capacity: int = 1000, max_rejections: int = 2000):
        self.capacity = capacity
        self.max_rejections = max_rejections
        
    def analyze_strategy(
        self,
        strategy: AbstractStrategy,
        scenario: Scenario,
        num_runs: int = 100,
        seeds: Optional[List[int]] = None,
        entity_generator_type: str = "multivariate"
    ) -> PerformanceMetrics:
        """Analyze a single strategy's performance."""
        
        if seeds is None:
            np.random.seed(42)
            seeds = np.random.randint(0, 1000000, num_runs).tolist()
        elif len(seeds) != num_runs:
            raise ValueError(f"Number of seeds ({len(seeds)}) must match num_runs ({num_runs})")
        
        results = []
        runtimes = []
        
        print(f"Analyzing {strategy.name} on {scenario.name} ({num_runs} runs)...")
        
        for i, seed in enumerate(seeds):
            if i % 10 == 0:
                print(f"  Progress: {i}/{num_runs}")
            
            # Run simulation with timing
            start_time = time.time()
            
            entity_generator = create_entity_generator(entity_generator_type, seed=seed)
            engine = SimulationEngine(entity_generator)
            result = engine.run_simulation(scenario, strategy, seed=seed)
            
            runtime = time.time() - start_time
            
            results.append(result)
            runtimes.append(runtime)
        
        return self._calculate_metrics(strategy.name, scenario, results, runtimes)
    
    def _calculate_metrics(
        self,
        strategy_name: str,
        scenario: Scenario,
        results: List[SimulationResult],
        runtimes: List[float]
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        # Basic arrays
        accepted_counts = [r.accepted_count for r in results]
        rejected_counts = [r.rejected_count for r in results]
        total_processed = [a + r for a, r in zip(accepted_counts, rejected_counts)]
        successes = [r.success for r in results]
        constraints_satisfied = [r.constraints_satisfied for r in results]
        
        # Basic statistics
        num_runs = len(results)
        success_rate = np.mean(successes)
        constraint_satisfaction_rate = np.mean(constraints_satisfied)
        
        avg_accepted = np.mean(accepted_counts)
        std_accepted = np.std(accepted_counts)
        avg_rejected = np.mean(rejected_counts)
        std_rejected = np.std(rejected_counts)
        
        # Acceptance rate
        total_accepted = sum(accepted_counts)
        total_processed_sum = sum(total_processed)
        acceptance_rate = total_accepted / total_processed_sum if total_processed_sum > 0 else 0
        
        # Capacity utilization
        capacity_utilization = avg_accepted / self.capacity
        capacity_utilization_std = std_accepted / self.capacity
        
        # Statistical metrics
        median_accepted = np.median(accepted_counts)
        min_accepted = min(accepted_counts)
        max_accepted = max(accepted_counts)
        
        # Efficiency metrics
        avg_entities_processed = np.mean(total_processed)
        rejection_efficiency = self._calculate_rejection_efficiency(results, scenario)
        
        # Runtime metrics
        avg_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        
        # Confidence intervals
        acceptance_ci = self._calculate_confidence_interval(accepted_counts)
        success_ci = self._calculate_proportion_confidence_interval(successes)
        
        # Constraint-specific metrics
        constraint_metrics = self._calculate_constraint_metrics(results, scenario)
        
        return PerformanceMetrics(
            strategy_name=strategy_name,
            scenario_name=scenario.name,
            num_runs=num_runs,
            success_rate=success_rate,
            constraint_satisfaction_rate=constraint_satisfaction_rate,
            avg_accepted=avg_accepted,
            std_accepted=std_accepted,
            avg_rejected=avg_rejected,
            std_rejected=std_rejected,
            acceptance_rate=acceptance_rate,
            capacity_utilization=capacity_utilization,
            capacity_utilization_std=capacity_utilization_std,
            constraint_metrics=constraint_metrics,
            median_accepted=median_accepted,
            min_accepted=min_accepted,
            max_accepted=max_accepted,
            avg_entities_processed=avg_entities_processed,
            rejection_efficiency=rejection_efficiency,
            avg_runtime_seconds=avg_runtime,
            std_runtime_seconds=std_runtime,
            acceptance_ci_95=acceptance_ci,
            success_rate_ci_95=success_ci
        )
    
    def _calculate_constraint_metrics(
        self,
        results: List[SimulationResult],
        scenario: Scenario
    ) -> Dict[str, Dict[str, float]]:
        """Calculate detailed constraint satisfaction metrics."""
        
        constraint_metrics = {}
        
        for constraint in scenario.constraints:
            attr = constraint.attribute
            
            # Get percentages for this attribute across all runs
            percentages = []
            violations = []
            
            for result in results:
                if result.accepted_count > 0:
                    pct = result.final_attribute_percentages.get(attr, 0.0)
                    percentages.append(pct)
                    violations.append(pct < constraint.min_percentage)
                else:
                    percentages.append(0.0)
                    violations.append(True)
            
            constraint_metrics[attr] = {
                'required_percentage': constraint.min_percentage,
                'avg_achieved_percentage': np.mean(percentages),
                'std_achieved_percentage': np.std(percentages),
                'min_achieved_percentage': np.min(percentages),
                'max_achieved_percentage': np.max(percentages),
                'violation_rate': np.mean(violations),
                'avg_surplus': np.mean([max(0, p - constraint.min_percentage) for p in percentages]),
                'avg_deficit': np.mean([max(0, constraint.min_percentage - p) for p in percentages])
            }
        
        return constraint_metrics
    
    def _calculate_rejection_efficiency(
        self,
        results: List[SimulationResult],
        scenario: Scenario
    ) -> float:
        """Calculate how efficiently the strategy uses rejections."""
        
        # Efficiency = constraint satisfaction rate / rejection rate
        # Higher efficiency means better use of rejections to satisfy constraints
        
        total_entities = sum(r.accepted_count + r.rejected_count for r in results)
        total_rejected = sum(r.rejected_count for r in results)
        
        if total_rejected == 0:
            return 1.0  # Perfect efficiency (no rejections needed)
        
        rejection_rate = total_rejected / total_entities
        constraint_satisfaction_rate = np.mean([r.constraints_satisfied for r in results])
        
        return constraint_satisfaction_rate / rejection_rate if rejection_rate > 0 else 0.0
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for continuous data."""
        
        if len(data) <= 1:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
        
        return (float(mean - h), float(mean + h))
    
    def _calculate_proportion_confidence_interval(self, data: List[bool], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for proportions."""
        
        n = len(data)
        if n <= 1:
            return (0.0, 0.0)
        
        p = np.mean(data)
        z = stats.norm.ppf((1 + confidence) / 2.)
        
        # Wilson score interval (more robust than normal approximation)
        denominator = 1 + z**2 / n
        centre_adjusted_probability = (p + z**2 / (2 * n)) / denominator
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        lower = centre_adjusted_probability - z * adjusted_standard_deviation
        upper = centre_adjusted_probability + z * adjusted_standard_deviation
        
        return (float(max(0, lower)), float(min(1, upper)))


class PerformanceReporter:
    """Generate comprehensive performance reports."""
    
    def __init__(self):
        pass
    
    def create_summary_table(self, metrics_list: List[PerformanceMetrics]):
        """Create a summary table of key metrics."""
        
        print("="*120)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("="*120)
        
        header = f"{'Strategy':<20} {'Scenario':<15} {'Success':<8} {'Constraints':<12} {'Avg Accept':<12} {'Accept Rate':<12} {'Efficiency':<12} {'Runtime(ms)':<12} {'Runs':<6}"
        print(header)
        print("-"*120)
        
        for metrics in metrics_list:
            print(f"{metrics.strategy_name:<20} "
                  f"{metrics.scenario_name:<15} "
                  f"{metrics.success_rate:.1%}    "
                  f"{metrics.constraint_satisfaction_rate:.1%}        "
                  f"{metrics.avg_accepted:.0f}±{metrics.std_accepted:.0f}    "
                  f"{metrics.acceptance_rate:.1%}        "
                  f"{metrics.rejection_efficiency:.2f}         "
                  f"{metrics.avg_runtime_seconds*1000:.1f}         "
                  f"{metrics.num_runs}")
        
        print("-"*120)
    
    def create_detailed_constraint_table(self, metrics_list: List[PerformanceMetrics]):
        """Create detailed constraint satisfaction table."""
        
        print("\n* DETAILED CONSTRAINT ANALYSIS")
        print("-" * 80)
        
        for metrics in metrics_list:
            print(f"\n{metrics.strategy_name} ({metrics.scenario_name}):")
            for attr, constraint_data in metrics.constraint_metrics.items():
                required = constraint_data['required_percentage']
                achieved = constraint_data['avg_achieved_percentage']
                violation_rate = constraint_data['violation_rate']
                status = "✓" if violation_rate == 0 else f"✗({violation_rate:.1%})"
                
                print(f"  {attr}: {achieved:.1%} (req: {required:.1%}) {status}")
                if violation_rate > 0:
                    deficit = constraint_data['avg_deficit']
                    print(f"    Avg deficit: {deficit:.1%}")
                else:
                    surplus = constraint_data['avg_surplus']
                    if surplus > 0:
                        print(f"    Avg surplus: {surplus:.1%}")
    
    def create_statistical_summary(self, metrics_list: List[PerformanceMetrics]):
        """Create statistical summary with confidence intervals."""
        
        print("\n* STATISTICAL ANALYSIS")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Scenario':<15} {'Success Rate (95% CI)':<25} {'Accepted (95% CI)':<20} {'CV':<8}")
        print("-" * 80)
        
        for metrics in metrics_list:
            ci_low, ci_high = metrics.acceptance_ci_95
            success_ci_low, success_ci_high = metrics.success_rate_ci_95
            cv = f"{metrics.std_accepted/metrics.avg_accepted:.2f}" if metrics.avg_accepted > 0 else "N/A"
            
            print(f"{metrics.strategy_name:<20} "
                  f"{metrics.scenario_name:<15} "
                  f"{metrics.success_rate:.1%} [{success_ci_low:.1%}, {success_ci_high:.1%}]   "
                  f"{metrics.avg_accepted:.0f} [{ci_low:.0f}, {ci_high:.0f}]    "
                  f"{cv}")
        
        print("-" * 80)
    
    def print_comprehensive_report(self, metrics_list: List[PerformanceMetrics]):
        """Print a comprehensive performance report."""
        
        print("="*80)
        print("* STRATEGY PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Summary table
        self.create_summary_table(metrics_list)
        
        # Statistical summary
        self.create_statistical_summary(metrics_list)
        
        # Constraint analysis
        self.create_detailed_constraint_table(metrics_list)
        
        # Performance insights
        print("\n* PERFORMANCE INSIGHTS")
        print("-" * 50)
        self._print_insights(metrics_list)
    
    def _print_insights(self, metrics_list: List[PerformanceMetrics]):
        """Print key insights from the analysis."""
        
        # Best performing strategy
        best_success = max(metrics_list, key=lambda m: m.success_rate)
        best_efficiency = max(metrics_list, key=lambda m: m.rejection_efficiency)
        best_utilization = max(metrics_list, key=lambda m: m.capacity_utilization)
        
        print(f"Highest Success Rate: {best_success.strategy_name} ({best_success.success_rate:.1%})")
        print(f"Most Efficient Rejections: {best_efficiency.strategy_name} ({best_efficiency.rejection_efficiency:.2f})")
        print(f"Best Capacity Utilization: {best_utilization.strategy_name} ({best_utilization.capacity_utilization:.1%})")
        
        # Consistency analysis
        most_consistent = min(metrics_list, key=lambda m: m.std_accepted / m.avg_accepted if m.avg_accepted > 0 else float('inf'))
        print(f"Most Consistent: {most_consistent.strategy_name} (CV: {most_consistent.std_accepted/most_consistent.avg_accepted:.2f})")
        
        # Speed analysis
        fastest = min(metrics_list, key=lambda m: m.avg_runtime_seconds)
        print(f"Fastest Runtime: {fastest.strategy_name} ({fastest.avg_runtime_seconds*1000:.1f}ms avg)")
        
        print("="*80)
