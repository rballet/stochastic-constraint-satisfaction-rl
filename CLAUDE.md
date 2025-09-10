# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains implementations of reinforcement learning approaches for **stochastic constraint satisfaction problems**. These are optimization problems where:
- Decisions must be made sequentially under uncertainty
- Multiple constraints must be satisfied simultaneously
- Future information is not available when making decisions
- The goal is to optimize objective functions while meeting all constraints

## Architecture

The codebase is organized into several key modules:

### Core Framework (`src/core/`)
- `types.py`: Core data structures (Problem, State, Constraint, Scenario, Result)
- `strategy_base.py`: Abstract base class for decision strategies with built-in optimization
- `environment_base.py`: Base class for stochastic CSP environments
- `constraint_manager.py`: Constraint validation and management utilities

### Strategy Implementations (`src/strategies/`)
- `greedy.py`: Simple greedy acceptance strategies
- `lp_strategy.py`: Linear programming optimization approaches  
- `dp_strategy.py`: Dynamic programming solutions
- `adaptive_dual.py`: Adaptive dual approaches with resource allocation
- `mathematical_optimal.py`: Mathematical optimization strategies
- `primal_dual.py`: Primal-dual optimization algorithms
- `constraint_optimized.py`: Constraint-aware optimization strategies

### Reinforcement Learning (`src/rl/`)
- `base_env.py`: Base Gymnasium environment for stochastic CSP
- `train_ppo.py`, `train_a2c.py`, `train_dqn.py`: RL algorithm training scripts
- `safe_env.py`: Safety-constrained RL environment wrapper
- `constrained_algorithms.py`: Constrained RL implementations
- Various training scripts for different RL approaches with safety constraints

### Example Problems (`src/examples/`)
- `hospital_icu/`: Hospital ICU Admission implementation
  - `icu_env.py`: Gymnasium environment for ICU bed allocation
  - `patient_generator.py`: Patient generation with medical attribute distributions
  - `icu_strategies.py`: Problem-specific medical allocation strategies
- `resource_allocation/`: Resource allocation under uncertainty examples
- `scheduling/`: Stochastic scheduling problems
- `portfolio/`: Portfolio optimization with constraints

### Simulation Engine (`src/simulation/`)
- `engine.py`: Main simulation runner with detailed logging
- `monte_carlo.py`: Monte Carlo simulation utilities
- `evaluation.py`: Performance evaluation and metrics

## Common Development Commands

### Testing Strategies
```bash
# Test strategies on different scenarios
python run_example.py --problem hospital_icu --strategy lp --scenario 2
python run_example.py --problem resource_allocation --strategy greedy --scenario 1

# Run Monte Carlo simulation
python run_monte_carlo.py --problem hospital_icu --runs 1000

# Compare multiple strategies
python compare_strategies.py --problem hospital_icu --strategies lp,dp,greedy
```

### RL Training and Evaluation
```bash
# Train PPO on Hospital ICU problem
python -m src.rl.train_ppo --problem hospital_icu --scenario 2 --timesteps 500000 --n_envs 8 --save models/ppo_icu_s2

# Train A2C (baseline)
python -m src.rl.train_a2c --problem hospital_icu --scenario 2 --timesteps 400000 --n_envs 8 --save models/a2c_icu_s2

# Debug/evaluate RL model with diagnostics
python -m debug.debug_rl --problem hospital_icu --scenario 2 --runs 3 --model models/ppo_icu_s2.zip

# Train constrained RL approaches
python train_constrained.py --problem hospital_icu --algorithm pcpo
python train_safe_rl.py --problem resource_allocation --algorithm cpo
```

### Code Quality
```bash
# Format code
black .

# Lint code  
flake8 .

# Type checking
mypy src/
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific problem tests
pytest tests/test_hospital_icu.py
pytest tests/test_resource_allocation.py
```

### Package Management
```bash
# Install dependencies
pip install -r requirements.txt

# Use virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Key Development Patterns

### Implementing New Problems
1. Create problem-specific directory in `src/examples/`
2. Inherit from `BaseStochasticCSPEnvironment` 
3. Define problem-specific state, action, and constraint structures
4. Implement reward shaping and termination conditions

### Implementing New Strategies
1. Inherit from `AbstractStrategy` in `src/core/strategy_base.py`
2. Implement `_decide_impl()` method (not `decide()` - base class handles optimization)
3. Implement `_reset_internal_state()` for state management
4. The base class automatically optimizes once constraints are satisfied

### Strategy Optimization
All strategies inherit automatic optimization: once all constraints are satisfied with sufficient margins, the base class can automatically switch to optimization mode to maximize the objective function.

### Constraint Management
- Constraints are defined as functions of the current state
- Support for both hard constraints (must be satisfied) and soft constraints (penalties)
- Constraint violations are tracked and used in reward shaping
- Dynamic constraint activation based on problem state

### RL Environment Details  
- Uses Gymnasium interface with consistent observation/action spaces
- Observation includes problem state, entity attributes, constraint status
- Action space varies by problem (binary, discrete, continuous)
- Reward shaping includes feasibility awareness and constraint penalties
- Models saved as `.zip` files in `models/` directory with naming convention

## File Naming Conventions
- `run_*.py`: Strategy execution scripts for specific problems/scenarios
- `train_*.py`: RL training scripts  
- `debug_*.py`: Debugging and diagnostic tools
- `compare_*.py`: Strategy comparison utilities
- Strategy files use descriptive names (e.g., `lp_strategy.py`, `adaptive_dual.py`)
- Models saved with format: `{algorithm}_{problem}_{scenario}_{variant}.zip`

## Important Implementation Notes
- Constraints must be evaluated on the current state (not historical data)
- Problems end when capacity is reached OR rejection limit is hit OR time limit exceeded
- Entity generation uses configurable distributions and correlations
- RL training benefits from obs_version=2 with feasibility-aware reward shaping
- The base strategy class handles optimization automatically - strategies focus on constraint satisfaction
- All problems should support Monte Carlo evaluation for benchmarking
- Constraint violations should be differentiable for gradient-based methods