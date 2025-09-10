"""
Monte Carlo simulation framework for stochastic constraint satisfaction problems.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import json
from pathlib import Path

from ..core.types import Scenario, SimulationResult
from ..core.strategy_base import AbstractStrategy
from .engine import SimulationEngine
from .entity_generator import EntityGenerator, create_entity_generator


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    num_runs: int = 1000
    num_parallel_workers: Optional[int] = None  # None = use all available cores
    random_seeds: Optional[List[int]] = None  # None = generate random seeds
    entity_generator_type: str = "correlated"
    log_individual_runs: bool = False
    save_results: bool = True
    results_directory: str = "monte_carlo_results"
    progress_callback: Optional[Callable[[int, int], None]] = None


@dataclass 
class StrategyPerformance:
    """Performance metrics for a strategy across multiple runs."""
    strategy_name: str
    scenario_name: str
    num_runs: int
    
    # Success metrics
    success_count: int
    success_rate: float
    
    # Rejection statistics
    rejections_mean: float
    rejections_std: float
    rejections_min: int
    rejections_max: int
    rejections_median: float
    rejections_p25: float
    rejections_p75: float
    rejections_p95: float
    
    # Acceptance statistics  
    accepted_mean: float
    accepted_std: float
    
    # Constraint satisfaction analysis
    constraint_violations: Dict[str, int] = field(default_factory=dict)
    
    # Runtime statistics
    total_runtime_seconds: float = 0.0
    average_runtime_per_simulation: float = 0.0
    
    # Raw data (optional, for detailed analysis)
    all_rejection_counts: List[int] = field(default_factory=list)
    all_success_flags: List[bool] = field(default_factory=list)
    detailed_results: List[SimulationResult] = field(default_factory=list)
    # Empirical arrival stats aggregated across runs
    arrival_total: int = 0
    arrival_attribute_probs: Dict[str, float] = field(default_factory=dict)
    arrival_pair_probs: Dict[Tuple[str, str], float] = field(default_factory=dict)


@dataclass
class MonteCarloResults:
    """Complete results from a Monte Carlo study."""
    config: MonteCarloConfig
    scenario: Scenario
    strategy_performances: List[StrategyPerformance]
    total_runtime_seconds: float
    timestamp: str
    
    def get_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance for specific strategy."""
        for perf in self.strategy_performances:
            if perf.strategy_name == strategy_name:
                return perf
        return None
    
    def get_ranking(self, rank_by: str = "success_rate") -> List[StrategyPerformance]:
        """Get strategies ranked by performance metric."""
        if rank_by == "success_rate":
            return sorted(self.strategy_performances, 
                         key=lambda x: (x.success_rate, -x.rejections_mean), 
                         reverse=True)
        elif rank_by == "rejections":
            successful_strategies = [p for p in self.strategy_performances if p.success_rate > 0]
            return sorted(successful_strategies, key=lambda x: x.rejections_mean)
        else:
            raise ValueError(f"Unknown ranking metric: {rank_by}")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for serialization."""
        return {
            "scenario_name": self.scenario.name,
            "num_strategies": len(self.strategy_performances),
            "total_simulations": sum(p.num_runs for p in self.strategy_performances),
            "total_runtime_seconds": self.total_runtime_seconds,
            "timestamp": self.timestamp,
            "config": {
                "num_runs": self.config.num_runs,
                "entity_generator_type": self.config.entity_generator_type
            },
            "strategy_summary": [
                {
                    "name": p.strategy_name,
                    "success_rate": p.success_rate,
                    "mean_rejections": p.rejections_mean,
                    "std_rejections": p.rejections_std
                }
                for p in self.strategy_performances
            ]
        }


class MonteCarloSimulator:
    """Monte Carlo simulator for testing strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def run_study(
        self, 
        scenario: Scenario,
        strategies: List[AbstractStrategy],
        config: MonteCarloConfig
    ) -> MonteCarloResults:
        """
        Run a complete Monte Carlo study comparing multiple strategies.
        
        Args:
            scenario: The scenario to test on
            strategies: List of strategies to compare
            config: Monte Carlo configuration
            
        Returns:
            Complete results from the study
        """
        start_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Starting Monte Carlo study: {len(strategies)} strategies, "
                        f"{config.num_runs} runs each")
        
        # Generate seeds if not provided
        if config.random_seeds is None:
            np.random.seed(42)  # For reproducible seed generation
            config.random_seeds = np.random.randint(0, 1000000, config.num_runs).tolist()
        
        # Run simulations for each strategy
        strategy_performances = []
        
        for i, strategy in enumerate(strategies):
            self.logger.info(f"Testing strategy {i+1}/{len(strategies)}: {strategy.name}")
            
            if config.progress_callback:
                config.progress_callback(i * config.num_runs, len(strategies) * config.num_runs)
            
            performance = self._run_strategy_simulations(
                scenario, strategy, config, config.random_seeds
            )
            strategy_performances.append(performance)
        
        total_runtime = time.time() - start_time
        
        # Create results
        results = MonteCarloResults(
            config=config,
            scenario=scenario,
            strategy_performances=strategy_performances,
            total_runtime_seconds=total_runtime,
            timestamp=timestamp
        )
        
        # Save results if requested
        if config.save_results:
            self._save_results(results)
        
        self.logger.info(f"Monte Carlo study completed in {total_runtime:.2f} seconds")
        return results
    
    def _run_strategy_simulations(
        self,
        scenario: Scenario,
        strategy: AbstractStrategy,
        config: MonteCarloConfig,
        seeds: List[int]
    ) -> StrategyPerformance:
        """Run simulations for a single strategy."""
        
        strategy_start_time = time.time()
        
        # Create entity generator
        entity_generator = create_entity_generator(config.entity_generator_type)
        engine = SimulationEngine(entity_generator, self.logger)
        
        # Run simulations
        if config.num_parallel_workers and config.num_parallel_workers > 1:
            results = self._run_parallel_simulations(
                scenario, strategy, engine, seeds, config
            )
        else:
            results = self._run_sequential_simulations(
                scenario, strategy, engine, seeds, config
            )
        
        strategy_runtime = time.time() - strategy_start_time
        
        # Analyze results
        performance = self._analyze_strategy_performance(
            strategy.name, scenario.name, results, strategy_runtime
        )
        
        if config.log_individual_runs:
            performance.detailed_results = results
        
        return performance
    
    def _run_sequential_simulations(
        self,
        scenario: Scenario,
        strategy: AbstractStrategy,
        engine: SimulationEngine,
        seeds: List[int],
        config: MonteCarloConfig
    ) -> List[SimulationResult]:
        """Run simulations sequentially."""
        
        results = []
        for i, seed in enumerate(seeds):
            if config.progress_callback and i % 100 == 0:
                config.progress_callback(i, len(seeds))
            
            result = engine.run_simulation(
                scenario, strategy, seed=seed, log_decisions=config.log_individual_runs
            )
            results.append(result)
        
        return results
    
    def _run_parallel_simulations(
        self,
        scenario: Scenario,
        strategy: AbstractStrategy,
        engine: SimulationEngine,
        seeds: List[int],
        config: MonteCarloConfig
    ) -> List[SimulationResult]:
        """Run simulations in parallel."""
        
        # Note: This is a placeholder for parallel execution
        # In practice, you'd need to handle serialization of objects
        # For now, fall back to sequential
        self.logger.warning("Parallel execution not yet implemented, using sequential")
        return self._run_sequential_simulations(scenario, strategy, engine, seeds, config)
    
    def _analyze_strategy_performance(
        self,
        strategy_name: str,
        scenario_name: str,
        results: List[SimulationResult],
        runtime_seconds: float
    ) -> StrategyPerformance:
        """Analyze performance from simulation results."""
        
        # Extract metrics
        successes = [r.success for r in results]
        rejections = [r.rejected_count for r in results]
        accepted = [r.accepted_count for r in results]
        
        # Success metrics
        success_count = sum(successes)
        success_rate = success_count / len(results)
        
        # Rejection statistics
        rejections_array = np.array(rejections)
        
        # Acceptance statistics
        accepted_array = np.array(accepted)
        
        # Constraint violation analysis
        constraint_violations = {}
        for result in results:
            if not result.constraints_satisfied:
                # This is simplified - in practice you'd want to track which specific constraints failed
                constraint_violations["any_constraint"] = constraint_violations.get("any_constraint", 0) + 1

        # Aggregate empirical arrival stats
        total_arrivals = sum(r.arrival_total for r in results)
        attr_counts: Dict[str, int] = {}
        pair_counts: Dict[Tuple[str, str], int] = {}
        for r in results:
            for a, c in r.arrival_attribute_counts.items():
                attr_counts[a] = attr_counts.get(a, 0) + c
            for ai, inner in r.arrival_pair_counts.items():
                for aj, c in inner.items():
                    key = (ai, aj)
                    pair_counts[key] = pair_counts.get(key, 0) + c

        arrival_attribute_probs = {a: (attr_counts[a] / total_arrivals) if total_arrivals > 0 else 0.0 for a in attr_counts}
        arrival_pair_probs = {k: (pair_counts[k] / total_arrivals) if total_arrivals > 0 else 0.0 for k in pair_counts}
        
        return StrategyPerformance(
            strategy_name=strategy_name,
            scenario_name=scenario_name,
            num_runs=len(results),
            success_count=success_count,
            success_rate=success_rate,
            rejections_mean=float(np.mean(rejections_array)),
            rejections_std=float(np.std(rejections_array)),
            rejections_min=int(np.min(rejections_array)),
            rejections_max=int(np.max(rejections_array)),
            rejections_median=float(np.median(rejections_array)),
            rejections_p25=float(np.percentile(rejections_array, 25)),
            rejections_p75=float(np.percentile(rejections_array, 75)),
            rejections_p95=float(np.percentile(rejections_array, 95)),
            accepted_mean=float(np.mean(accepted_array)),
            accepted_std=float(np.std(accepted_array)),
            constraint_violations=constraint_violations,
            total_runtime_seconds=runtime_seconds,
            average_runtime_per_simulation=runtime_seconds / len(results),
            all_rejection_counts=rejections,
            all_success_flags=successes,
            arrival_total=total_arrivals,
            arrival_attribute_probs=arrival_attribute_probs,
            arrival_pair_probs=arrival_pair_probs
        )
    
    def _save_results(self, results: MonteCarloResults) -> None:
        """Save results to file."""
        
        # Create results directory
        results_dir = Path(results.config.results_directory)
        results_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary_file = results_dir / f"summary_{results.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results.to_summary_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to {summary_file}")


def create_default_config(num_runs: int = 1000) -> MonteCarloConfig:
    """Create default Monte Carlo configuration."""
    return MonteCarloConfig(num_runs=num_runs)


def run_quick_comparison(
    scenario: Scenario,
    strategies: List[AbstractStrategy],
    num_runs: int = 100
) -> MonteCarloResults:
    """Run a quick comparison of strategies."""
    
    config = create_default_config(num_runs)
    config.save_results = False
    config.log_individual_runs = False
    
    simulator = MonteCarloSimulator()
    return simulator.run_study(scenario, strategies, config)