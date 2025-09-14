#!/usr/bin/env python3
"""
DQN Training Script for ICU Admission Problem

This script provides comprehensive training, evaluation, and model management
for DQN agents solving the ICU admission constraint satisfaction problem.

Features:
- Multiple ICU scenarios (standard, high-acuity, emergency)
- Hyperparameter optimization
- Training progress monitoring
- Model saving and loading
- Performance evaluation and comparison
- Visualization of training metrics

Usage:
    python src/examples/hospital_icu/train_dqn.py --scenario standard --timesteps 50000
    python src/examples/hospital_icu/train_dqn.py --scenario all --timesteps 100000 --curriculum
    python src/examples/hospital_icu/train_dqn.py --scenario all --timesteps 100000 --optimize
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

from src.strategies.dqn_strategy import DQNStrategy, DQNConfig, SCSPEnvironment
from src.examples.hospital_icu.scenarios import (
    create_icu_scenario_1, create_icu_scenario_2, create_icu_scenario_3, create_icu_scenario_4
)
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.simulation.engine import SimulationEngine


class ICUDQNTrainer:
    """Comprehensive DQN trainer for ICU admission problem."""
    
    def __init__(self, output_dir: str = "models/icu_dqn"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available scenarios
        self.scenarios = {
            'standard': ("ICU Standard", create_icu_scenario_1),
            'high_acuity': ("ICU High-Acuity", create_icu_scenario_2),
            'emergency': ("ICU Emergency", create_icu_scenario_3),
            'negative_correlations': ("ICU Negative Correlations", create_icu_scenario_4)
        }
        
        self.trained_models = {}
        self.training_histories = {}
        
    def create_training_config(self, 
                              total_timesteps: int = 100000,
                              learning_rate: float = 0.0003,
                              buffer_size: int = 200000,
                              exploration_fraction: float = 0.4,
                              target_update_interval: int = 2000,
                              eval_freq: int = 10000) -> DQNConfig:
        """Create optimized DQN configuration for ICU problem."""
        
        return DQNConfig(
            # Core DQN parameters
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=target_update_interval,
            
            # Exploration schedule
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            
            # Enhanced network architecture for complex constraint reasoning
            net_arch=[512, 256, 128, 64],  # Deeper network for complex constraint satisfaction
            
            # Advanced features
            optimize_memory_usage=False,
            # Note: SB3's standard DQN doesn't support prioritized replay
            
            # Enhanced ICU-specific reward tuning for better learning
            capacity_reward_weight=0.2,
            constraint_reward_weight=5.0,  # Stronger constraint satisfaction rewards
            terminal_reward_scale=20.0,    # Even stronger terminal incentives
            dilution_penalty_weight=0.1,   # More significant dilution penalties
            
            # Training configuration
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=30,  # More evaluation episodes for better statistics
            verbose=1,
            seed=42
        )
    
    def train_scenario(self, scenario_key: str, config: DQNConfig) -> DQNStrategy:
        """Train DQN agent on a specific ICU scenario."""
        
        if scenario_key not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_key}")
        
        scenario_name, scenario_factory = self.scenarios[scenario_key]
        scenario = scenario_factory()
        
        print(f"\n🏥 Training DQN on {scenario_name}")
        print(f"   Timesteps: {config.total_timesteps:,}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Network: {config.net_arch}")
        print("=" * 60)
        
        # Create and train strategy with unified state space
        strategy = DQNStrategy(scenario, config)
        
        start_time = time.time()
        training_history = strategy.train_agent()
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.1f}s")
        
        # Save model
        model_path = self.output_dir / f"dqn_{scenario_key}_{config.total_timesteps}.zip"
        strategy.save_model(str(model_path))
        print(f"📁 Model saved: {model_path}")
        
        # Save configuration
        config_path = self.output_dir / f"config_{scenario_key}_{config.total_timesteps}.json"
        with open(config_path, 'w') as f:
            config_dict = asdict(config)
            config_dict['scenario'] = scenario_key
            config_dict['training_time'] = training_time
            json.dump(config_dict, f, indent=2)
        
        # Store results
        self.trained_models[scenario_key] = strategy
        self.training_histories[scenario_key] = training_history
        
        return strategy
    
    def train_with_curriculum(self, scenario_keys: List[str], config: DQNConfig, 
                             curriculum_ratios: List[float] = None) -> DQNStrategy:
        """Train DQN with curriculum learning across multiple scenarios."""
        
        if curriculum_ratios is None:
            # Default: start with easier scenarios, progress to harder ones
            curriculum_ratios = [0.1, 0.15, 0.25, 0.5]  # Adjust based on scenario difficulty
        
        if len(curriculum_ratios) != len(scenario_keys):
            curriculum_ratios = [1.0 / len(scenario_keys)] * len(scenario_keys)
        
        print(f"\n🎓 Starting curriculum training across {len(scenario_keys)} scenarios")
        print(f"   Total timesteps: {config.total_timesteps:,}")
        print(f"   Curriculum ratios: {curriculum_ratios}")
        
        # Use the first scenario for initial setup
        primary_scenario = self.scenarios[scenario_keys[0]][1]()
        strategy = DQNStrategy(primary_scenario, config)
        
        # Train in stages
        for stage, (scenario_key, ratio) in enumerate(zip(scenario_keys, curriculum_ratios)):
            stage_timesteps = int(config.total_timesteps * ratio)
            if stage_timesteps == 0:
                continue
                
            scenario_name, scenario_factory = self.scenarios[scenario_key]
            scenario = scenario_factory()
            
            print(f"\n📚 Stage {stage + 1}: Training on {scenario_name}")
            print(f"   Timesteps for this stage: {stage_timesteps:,}")
            
            # Update environment for this scenario
            strategy.env = SCSPEnvironment(scenario, config)
            strategy.scenario = scenario
            
            # Create new model if this is the first stage, otherwise continue training
            if stage == 0:
                strategy._create_model()
            else:
                # Update the model's environment
                strategy.model.set_env(strategy.env)
            
            # Train on this scenario
            temp_config = DQNConfig(**asdict(config))
            temp_config.total_timesteps = stage_timesteps
            temp_config.verbose = 0 if stage > 0 else 1  # Reduce verbosity for later stages
            
            strategy.train_agent(total_timesteps=stage_timesteps)
            
            print(f"   ✅ Stage {stage + 1} completed")
        
        # Final evaluation on the target scenario (last in the list)
        target_scenario = self.scenarios[scenario_keys[-1]][1]()
        strategy.env = SCSPEnvironment(target_scenario, config)
        strategy.scenario = target_scenario
        strategy.model.set_env(strategy.env)
        
        # Store the final model with just the target scenario name
        target_scenario_key = scenario_keys[-1]
        self.trained_models[target_scenario_key] = strategy

        # Save the final curriculum-trained model
        model_path = self.output_dir / f"dqn_curriculum_{target_scenario.category}_{config.total_timesteps}.zip"
        strategy.save_model(str(model_path))
        print(f"📁 Curriculum model saved: {model_path}")

        # Save configuration with curriculum information
        config_path = self.output_dir / f"config_curriculum_{target_scenario.category}_{config.total_timesteps}.json"
        with open(config_path, 'w') as f:
            config_dict = asdict(config)
            config_dict['scenario'] = target_scenario_key
            config_dict['curriculum_scenarios'] = scenario_keys
            config_dict['curriculum_ratios'] = curriculum_ratios
            config_dict['training_method'] = 'curriculum'
            json.dump(config_dict, f, indent=2)

        print(f"\n🎯 Curriculum training completed. Final model ready for {target_scenario_key}")
        return strategy
    
    def evaluate_trained_model(self, scenario_key: str, num_episodes: int = 100) -> Dict:
        """Evaluate a trained model comprehensively."""
        
        if scenario_key not in self.trained_models:
            raise ValueError(f"No trained model for scenario: {scenario_key}")
        
        strategy = self.trained_models[scenario_key]
        scenario_name, scenario_factory = self.scenarios[scenario_key]
        scenario = scenario_factory()
        
        print(f"\n📊 Evaluating {scenario_name} model...")
        
        # SB3 evaluation
        sb3_results = strategy.evaluate_agent(num_episodes=num_episodes)
        
        # Detailed simulation analysis using PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze_strategy(strategy, scenario, num_runs=num_episodes)
        
        # Combine results
        evaluation = {
            'scenario': scenario_key,
            'scenario_name': scenario_name,
            'episodes': num_episodes,
            'sb3_metrics': sb3_results,
            'detailed_metrics': {
                'success_rate': metrics.success_rate,
                'constraint_satisfaction_rate': metrics.constraint_satisfaction_rate,
                'avg_accepted': metrics.avg_accepted,
                'avg_rejected': metrics.avg_rejected,
                'acceptance_rate': metrics.acceptance_rate,
                'rejection_efficiency': metrics.rejection_efficiency,
                'avg_runtime_seconds': metrics.avg_runtime_seconds,
                'constraint_metrics': metrics.constraint_metrics
            }
        }
        
        print(f"   Success Rate: {evaluation['detailed_metrics']['success_rate']:.1%}")
        print(f"   Constraint Satisfaction: {evaluation['detailed_metrics']['constraint_satisfaction_rate']:.1%}")
        print(f"   Acceptance Rate: {evaluation['detailed_metrics']['acceptance_rate']:.1%}")
        print(f"   Rejection Efficiency: {evaluation['detailed_metrics']['rejection_efficiency']:.2f}")
        
        return evaluation
    
    def plot_training_history(self, scenario_key: str, save_plot: bool = True):
        """Plot training metrics for a scenario."""
        
        if scenario_key not in self.training_histories:
            raise ValueError(f"No training history for scenario: {scenario_key}")
        
        history = self.training_histories[scenario_key]
        scenario_name = self.scenarios[scenario_key][0]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot constraint satisfaction rate
        if history['constraint_satisfaction_rates']:
            episodes = range(len(history['constraint_satisfaction_rates']))
            axes[0].plot(episodes, history['constraint_satisfaction_rates'], 'b-', alpha=0.7)
            
            # Add moving average
            if len(history['constraint_satisfaction_rates']) > 10:
                window = min(100, len(history['constraint_satisfaction_rates']) // 4)
                ma = np.convolve(history['constraint_satisfaction_rates'], 
                               np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(episodes)), ma, 'r-', linewidth=2, 
                           label=f'MA({window})')
                axes[0].legend()
        
        axes[0].set_title(f'{scenario_name}: Constraint Satisfaction Rate')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Constraint Satisfaction Rate')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Plot success rate
        if history['success_rates']:
            episodes = range(len(history['success_rates']))
            axes[1].plot(episodes, history['success_rates'], 'g-', alpha=0.7)
            
            # Add moving average
            if len(history['success_rates']) > 10:
                window = min(100, len(history['success_rates']) // 4)
                ma = np.convolve(history['success_rates'], 
                               np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(episodes)), ma, 'r-', linewidth=2,
                           label=f'MA({window})')
                axes[1].legend()
        
        axes[1].set_title(f'{scenario_name}: Success Rate')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Success Rate')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"training_history_{scenario_key}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 Training plot saved: {plot_path}")
        
        if save_plot:
            plt.close()  # Close figure to free memory
    
    def compare_scenarios(self, evaluation_results: List[Dict]):
        """Create comparison visualization across scenarios."""
        
        scenario_names = [result['scenario_name'] for result in evaluation_results]
        success_rates = [result['detailed_metrics']['success_rate'] for result in evaluation_results]
        constraint_rates = [result['detailed_metrics']['constraint_satisfaction_rate'] for result in evaluation_results]
        acceptance_rates = [result['detailed_metrics']['acceptance_rate'] for result in evaluation_results]
        efficiencies = [result['detailed_metrics']['rejection_efficiency'] for result in evaluation_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success rates
        axes[0,0].bar(scenario_names, success_rates, color='green', alpha=0.7)
        axes[0,0].set_title('Success Rate by Scenario')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].set_ylim(0, 1)
        
        # Constraint satisfaction rates
        axes[0,1].bar(scenario_names, constraint_rates, color='blue', alpha=0.7)
        axes[0,1].set_title('Constraint Satisfaction Rate by Scenario')
        axes[0,1].set_ylabel('Constraint Satisfaction Rate')
        axes[0,1].set_ylim(0, 1)
        
        # Acceptance rates
        axes[1,0].bar(scenario_names, acceptance_rates, color='orange', alpha=0.7)
        axes[1,0].set_title('Acceptance Rate by Scenario')
        axes[1,0].set_ylabel('Acceptance Rate')
        axes[1,0].set_ylim(0, 1)
        
        # Rejection Efficiency
        axes[1,1].bar(scenario_names, efficiencies, color='purple', alpha=0.7)
        axes[1,1].set_title('Rejection Efficiency by Scenario')
        axes[1,1].set_ylabel('Rejection Efficiency')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "scenario_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved: {plot_path}")
        plt.close()  # Close figure to free memory
    
    def hyperparameter_search(self, scenario_key: str, 
                            total_timesteps: int = 25000,
                            n_trials: int = 5) -> Dict:
        """Simple hyperparameter search for a scenario."""
        
        print(f"\n🔍 Hyperparameter search for {scenario_key} ({n_trials} trials)")
        
        # Define search space
        learning_rates = [0.0001, 0.0005, 0.001]
        exploration_fractions = [0.2, 0.3, 0.4]
        constraint_weights = [2.0, 3.0, 5.0]
        
        best_score = 0
        best_config = None
        results = []
        
        trial = 0
        for lr in learning_rates:
            for exp_frac in exploration_fractions:
                for constraint_weight in constraint_weights:
                    if trial >= n_trials:
                        break
                    
                    print(f"\nTrial {trial + 1}/{n_trials}: lr={lr}, exp_frac={exp_frac}, cw={constraint_weight}")
                    
                    config = self.create_training_config(
                        total_timesteps=total_timesteps,
                        learning_rate=lr,
                        exploration_fraction=exp_frac
                    )
                    config.constraint_reward_weight = constraint_weight
                    config.eval_freq = max(5000, total_timesteps // 5)
                    config.verbose = 0  # Reduce output during search
                    
                    strategy = self.train_scenario(f"{scenario_key}_trial_{trial}", config)
                    eval_result = self.evaluate_trained_model(f"{scenario_key}_trial_{trial}", num_episodes=50)
                    
                    # Score based on success rate and constraint satisfaction
                    score = (eval_result['detailed_metrics']['success_rate'] * 0.6 + 
                            eval_result['detailed_metrics']['constraint_satisfaction_rate'] * 0.4)
                    
                    results.append({
                        'trial': trial,
                        'config': asdict(config),
                        'score': score,
                        'evaluation': eval_result
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_config = config
                    
                    print(f"   Score: {score:.3f}")
                    trial += 1
        
        print(f"\n🏆 Best configuration (score: {best_score:.3f}):")
        print(f"   Learning Rate: {best_config.learning_rate}")
        print(f"   Exploration Fraction: {best_config.exploration_fraction}")
        print(f"   Constraint Weight: {best_config.constraint_reward_weight}")
        
        # Save hyperparameter search results
        search_path = self.output_dir / f"hyperparameter_search_{scenario_key}.json"
        with open(search_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return {'best_config': best_config, 'best_score': best_score, 'all_results': results}


def main():
    """Main training script entry point."""
    
    parser = argparse.ArgumentParser(description='Train DQN agents for ICU admission problem')
    parser.add_argument('--scenario', choices=['standard', 'high_acuity', 'emergency', 'negative_correlations', 'all'], 
                       default='standard', help='ICU scenario to train on')
    parser.add_argument('--timesteps', type=int, default=50000, 
                       help='Number of training timesteps')
    parser.add_argument('--learning-rate', type=float, default=0.0005, 
                       help='Learning rate for training')
    parser.add_argument('--eval-episodes', type=int, default=100, 
                       help='Number of episodes for evaluation')
    parser.add_argument('--output-dir', type=str, default='models/icu_dqn', 
                       help='Output directory for models and results')
    parser.add_argument('--optimize', action='store_true', 
                       help='Run hyperparameter optimization')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate training plots')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning across scenarios')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ICUDQNTrainer(output_dir=args.output_dir)
    
    if args.optimize:
        print("🔍 Running hyperparameter optimization...")
        scenario_key = args.scenario if args.scenario != 'all' else 'standard'
        search_results = trainer.hyperparameter_search(
            scenario_key, 
            total_timesteps=args.timesteps // 2,  # Shorter for search
            n_trials=8
        )
        config = search_results['best_config']
        config.total_timesteps = args.timesteps  # Use full timesteps for final training
    else:
        config = trainer.create_training_config(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate
        )
    
    # Train on specified scenario(s)
    scenarios_to_train = ['standard', 'high_acuity', 'emergency', 'negative_correlations'] if args.scenario == 'all' else [args.scenario]
    
    evaluation_results = []
    
    if args.curriculum and len(scenarios_to_train) > 1:
        # Use curriculum learning for multiple scenarios
        print("🎓 Using curriculum learning approach...")

        # Order scenarios from easiest to hardest: standard -> high_acuity -> emergency
        curriculum_order = ['standard', 'high_acuity', 'emergency', 'negative_correlations']
        curriculum_scenarios = [s for s in curriculum_order if s in scenarios_to_train]
        
        strategy = trainer.train_with_curriculum(curriculum_scenarios, config)
        
        # Evaluate on the final (hardest) scenario
        final_scenario = curriculum_scenarios[-1]
        eval_result = trainer.evaluate_trained_model(final_scenario, num_episodes=args.eval_episodes)
        evaluation_results.append(eval_result)
        
    else:
        # Traditional training approach
        print("🎓 Using traditional training approach...")
        for scenario_key in scenarios_to_train:
            # Train model
            strategy = trainer.train_scenario(scenario_key, config)
            
            # Evaluate model
            eval_result = trainer.evaluate_trained_model(scenario_key, num_episodes=args.eval_episodes)
            evaluation_results.append(eval_result)
            
            # Plot training history
            if args.plot:
                trainer.plot_training_history(scenario_key)
    
    # Create comparison if multiple scenarios
    if len(evaluation_results) > 1:
        trainer.compare_scenarios(evaluation_results)
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    for result in evaluation_results:
        print(f"\n{result['scenario_name']}:")
        metrics = result['detailed_metrics']
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
        print(f"  Constraint Satisfaction: {metrics['constraint_satisfaction_rate']:.1%}")
        print(f"  Acceptance Rate: {metrics['acceptance_rate']:.1%}")
        print(f"  Rejection Efficiency: {metrics['rejection_efficiency']:.2f}")
    
    print(f"\n📁 Models and results saved in: {trainer.output_dir}")
    
    if args.optimize:
        print(f"🔍 Best hyperparameters found:")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Exploration Fraction: {config.exploration_fraction}")
        print(f"   Constraint Weight: {config.constraint_reward_weight}")


if __name__ == "__main__":
    main()
