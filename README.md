# Stochastic Constraint Satisfaction with Reinforcement Learning

This repository contains implementations of reinforcement learning approaches for **stochastic constraint satisfaction problems**. These are optimization problems where decisions must be made sequentially under uncertainty while satisfying multiple constraints simultaneously.

## Project Overview

Stochastic constraint satisfaction problems involve:
- Sequential decision-making under uncertainty
- Multiple constraints that must be satisfied simultaneously
- Limited future information when making decisions
- Optimization of objective functions while meeting all constraints
- Constrained Markov Decision Process

For a comprehensive mathematical foundation and theoretical background, see [`docs/MATHEMATICAL_FOUNDATION.md`](docs/MATHEMATICAL_FOUNDATION.md).

## Example Problems

The repository includes various example problems:
- **Hospital ICU Admission**: ICU bed allocation with patient priority and medical constraints
- **Resource Allocation**: Dynamic resource allocation under uncertainty
- **Scheduling**: Stochastic scheduling with capacity constraints
- **Portfolio Optimization**: Portfolio construction with risk constraints

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stochastic-constraint-satisfaction-rl
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.core import Entity, Constraint, Scenario, ProblemState
from src.core.strategy_base import RandomStrategy
from src.simulation import SimulationEngine, create_entity_generator

# Define a hospital ICU admission scenario
scenario = Scenario(
    name="icu_admission",
    attributes=["critical_condition", "elderly", "has_insurance"],
    constraints=[
        Constraint(attribute="critical_condition", min_percentage=0.7, description="70% critical patients"),
        Constraint(attribute="elderly", min_percentage=0.4, description="40% elderly patients"),
    ],
    attribute_probabilities={"critical_condition": 0.5, "elderly": 0.3, "has_insurance": 0.8}
)

# Create strategy and run simulation
strategy = RandomStrategy(acceptance_rate=0.6)
generator = create_entity_generator("basic", seed=42)
engine = SimulationEngine(generator)

result = engine.run_simulation(scenario, strategy, seed=42)
print(f"Success: {result.success}, Accepted: {result.accepted_count}")
```

## Architecture

- **Core Framework**: Abstract types and strategy base classes
- **Strategies**: Various decision-making approaches (greedy, LP, DP, RL)
- **Simulation Engine**: Monte Carlo simulation and evaluation
- **RL Components**: Gymnasium environments and training scripts
- **Examples**: Specific problem implementations

## Documentation

- **[Mathematical Foundation](docs/MATHEMATICAL_FOUNDATION.md)**: Core theory and problem formulation
- **[Linear Programming Approach](docs/linear_programming_approach.md)**: LP-based solution methods and implementation
