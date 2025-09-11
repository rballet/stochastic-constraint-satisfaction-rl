# Linear Programming Approach to Stochastic Constraint Satisfaction

This document details how Linear Programming (LP) can be formulated and applied to solve Stochastic Constraint Satisfaction Problems (SCSP), providing both theoretical foundations and practical implementation strategies.

## Overview

Linear Programming offers a mathematically principled approach to SCSP by formulating the decision problem as an optimization problem with linear constraints. While SCSP is dynamic and stochastic, LP techniques can be adapted through approximation methods, lookahead strategies, and expected value formulations.

## LP Formulation

### Decision Variables

For each arriving entity $e_t$ at time $t$, we define:
$$x_t \in \{0, 1\}$$

Where:
- $x_t = 1$ if entity $e_t$ is accepted
- $x_t = 0$ if entity $e_t$ is rejected

### Objective Function

The LP formulation maximizes expected utility over a planning horizon:

$$\max \sum_{t=1}^T \mathbb{E}[R(s_t, e_t, x_t)]$$

Where $R(s_t, e_t, x_t)$ represents the reward function from the mathematical foundation.

### Constraint Framework

**1. Capacity Constraint:**
$$\sum_{t=1}^T x_t \leq N_{\max}$$

**2. Attribute Constraints:**
For each constraint $j$ requiring minimum percentage $\theta_j$ of attribute $a_j$:
$$\sum_{t=1}^T x_t \cdot \mathbf{1}_{e_t \text{ has } a_j} \geq \theta_j \cdot \sum_{t=1}^T x_t$$

**3. Non-negativity and Binary Constraints:**
$$x_t \in \{0, 1\} \quad \forall t$$

## Handling Stochasticity in LP

### Challenge: Unknown Future Arrivals

The fundamental challenge is that future entities $\{e_{t+1}, e_{t+2}, ..., e_T\}$ are unknown when making decision $x_t$.

### Solution

#### 1. Expected Value Formulation

The key insight is that we must distinguish between current entities (with known attributes) and future entities (with probabilistic attributes):

For the current entity $e_t$: attributes are deterministic $\mathbf{1}_{e_t \text{ has } a_j} \in \{0,1\}$
For future entities: use expected values $\mathbb{E}[e_{\tau} \text{ has attribute } a_j] = p_j$

This transforms stochastic constraints into deterministic ones by incorporating current state:
$$c_j^{(t)} + x_t \cdot \mathbf{1}_{e_t \text{ has } a_j} + y_{\text{future}} \cdot p_j \geq \theta_j \cdot (n_t + x_t + y_{\text{future}})$$

Where:
- $c_j^{(t)}$: count of already accepted entities with attribute $j$
- $n_t$: total already accepted entities
- $y_{\text{future}}$: expected future acceptances

#### 2. Lookahead Horizon Strategy

Instead of optimizing over the entire horizon $T$, use a rolling horizon approach:

At time $t$, solve:
$$\max \sum_{\tau=t}^{t+H} \mathbb{E}[R(s_\tau, e_\tau, x_\tau)]$$

Where $H$ is the lookahead horizon (e.g., next 50 entities).

#### 3. Conservative Buffer Approach

Add safety margins to constraints to account for uncertainty:
$$\sum_{t=1}^T x_t \cdot p_j \geq (\theta_j + \delta_j) \cdot \sum_{t=1}^T x_t$$

Where $\delta_j$ is a buffer parameter (e.g., 5% additional margin).

## Practical LP Implementation

### Current State Integration

At decision time $t$, incorporate current state $s_t$:

**Current Capacity Used:**
$$n_t = \sum_{\tau=1}^{t-1} x_\tau$$

**Current Attribute Counts:**
$$c_j^{(t)} = \sum_{\tau=1}^{t-1} x_\tau \cdot \mathbf{1}_{e_\tau \text{ has } a_j}$$

**Remaining Capacity:**
$$N_{\text{remaining}} = N_{\max} - n_t$$

### Future Entity Estimation

**Estimated Future Arrivals:**
Based on arrival process (e.g., Poisson with rate $\lambda$):
$$\mathbb{E}[\text{future entities}] = \lambda \cdot (T - t)$$

**Estimated Future Attributes:**
For attribute $j$ with probability $p_j$:
$$\mathbb{E}[\text{future entities with } a_j] = \lambda \cdot (T - t) \cdot p_j$$

### Acceptance Threshold Calculation

The LP solution provides an optimal acceptance strategy. For immediate decision-making, compute:

$$\text{Accept if: } \frac{\text{Marginal Benefit}}{\text{Marginal Cost}} > \text{Threshold}$$

Where:
- **Marginal Benefit**: Immediate reward + constraint progress
- **Marginal Cost**: Opportunity cost of using capacity
- **Threshold**: Derived from LP dual variables

## Mathematical Formulation Details

### Complete LP Model

Given current state $(n_t, c_1^{(t)}, ..., c_k^{(t)})$ and arriving entity $e_t$:

**Decision Variables:**
- $x_t \in \{0, 1\}$: Accept current entity
- $y_\tau \in [0, 1], \tau > t$: Expected acceptance of future entities

**Objective:**
$$\max \quad R_{\text{immediate}}(x_t) + \sum_{\tau=t+1}^{t+H} \mathbb{E}[R_{\text{future}}] \cdot y_\tau$$

**Constraints:**
$$\begin{align}
n_t + x_t + \sum_{\tau=t+1}^{t+H} y_\tau &\leq N_{\max} \\
c_j^{(t)} + x_t \cdot \mathbf{1}_{e_t \text{ has } a_j} + \sum_{\tau=t+1}^{t+H} y_\tau \cdot p_j &\geq \theta_j \cdot \left(n_t + x_t + \sum_{\tau=t+1}^{t+H} y_\tau\right) \\
0 \leq y_\tau &\leq 1 \quad \forall \tau > t \\
x_t &\in \{0, 1\}
\end{align}$$

This can be rearranged to highlight the decision impact:
$$c_j^{(t)} + x_t \cdot (\mathbf{1}_{e_t \text{ has } a_j} - \theta_j) + \sum_{\tau=t+1}^{t+H} y_\tau \cdot (p_j - \theta_j) \geq \theta_j \cdot n_t$$

The term $(\mathbf{1}_{e_t \text{ has } a_j} - \theta_j)$ shows when accepting entity $e_t$ helps:
- If entity has attribute: $1 - \theta_j > 0$ (helps if $\theta_j < 1$)  
- If entity lacks attribute: $0 - \theta_j < 0$ (hurts)

Similarly, $(p_j - \theta_j)$ indicates constraint feasibility:
- If $p_j > \theta_j$: future arrivals help this constraint on average
- If $p_j < \theta_j$: constraint is challenging but may still be satisfiable

## Implementation Strategy

### Algorithm Overview

```
1. Initialize LP solver and problem parameters
2. For each arriving entity e_t:
   a. Update current state (n_t, constraints)
   b. Estimate future entity characteristics
   c. Formulate LP with lookahead horizon
   d. Solve LP to get optimal x_t*
   e. Make decision based on x_t*
   f. Update state and move to next entity
```

### Key Parameters

**Configuration Parameters:**
- `acceptance_threshold`: Minimum LP objective value to accept (default: 0.5)
- `lookahead_horizon`: Number of future entities to consider (default: 50)
- `constraint_buffer`: Safety margin for constraints (default: 0.05)
- `use_conservative_estimates`: Whether to use pessimistic probability estimates

### Computational Considerations

**Solver Choice:**
- **Binary LP**: Use branch-and-bound for exact solutions
- **Relaxed LP**: Solve continuous relaxation for speed
- **Heuristic**: Round fractional solutions using intelligent rounding

**Performance:**
- **Time Complexity**: O(H × K) per decision, where H is horizon and K is constraints
- **Space Complexity**: O(H × K) for problem formulation
- **Scalability**: Suitable for moderate-sized problems (H ≤ 1000)

## Advantages and Limitations

### Advantages

1. **Mathematical Rigor**: Provides optimal solutions under modeling assumptions
2. **Interpretability**: Decision logic is transparent and explainable
3. **Constraint Handling**: Natural framework for hard constraints
4. **Sensitivity Analysis**: Dual variables provide marginal value insights
5. **Scalability**: Efficient solvers available for large problems

### Limitations

1. **Stochastic Approximation**: Replaces uncertainty with expected values
2. **Computational Cost**: May be slow for real-time decisions
3. **Model Accuracy**: Performance depends on accurate probability estimates
4. **Binary Decisions**: Integer programming is NP-hard
5. **Static Modeling**: Doesn't adapt to changing environments

## Comparison with Other Approaches

### vs. Greedy Strategies
- **LP Advantage**: Global optimization vs. local decisions
- **Greedy Advantage**: Speed and simplicity
- **Use Case**: LP when optimality matters, Greedy for speed

### vs. Dynamic Programming
- **LP Advantage**: Handles continuous spaces better
- **DP Advantage**: Exact treatment of uncertainty
- **Use Case**: LP for large state spaces, DP for small discrete spaces

### vs. Reinforcement Learning
- **LP Advantage**: No training required, immediate deployment
- **RL Advantage**: Learns from experience, adapts to environment
- **Use Case**: LP for interpretable policies, RL for complex environments

## Case Study: Hospital ICU Implementation

### Problem Setup
- **Capacity**: 100 ICU beds
- **Constraints**: 60% critical patients, 40% elderly patients
- **Arrival Rate**: 50 patients/day
- **Planning Horizon**: 200 patient arrivals

### LP Formulation

**Decision Variables:**
$$x_t \in \{0, 1\} \quad \text{(admit patient t)}$$

**Objective:**
$$\max \sum_{t=1}^{200} x_t \cdot \text{patient\_utility}_t - \text{constraint\_penalties}$$

**Constraints:**
At decision time $t$ with current state $(n_t, c_{\text{critical}}^{(t)}, c_{\text{elderly}}^{(t)})$:

$$\begin{align}
n_t + x_t + y_{\text{future}} &\leq 100 \\
c_{\text{critical}}^{(t)} + x_t \cdot \mathbf{1}_{\text{patient t is critical}} + y_{\text{future}} \cdot p_{\text{critical}} &\geq 0.6 \cdot (n_t + x_t + y_{\text{future}}) \\
c_{\text{elderly}}^{(t)} + x_t \cdot \mathbf{1}_{\text{patient t is elderly}} + y_{\text{future}} \cdot p_{\text{elderly}} &\geq 0.4 \cdot (n_t + x_t + y_{\text{future}})
\end{align}$$

Where:
- $p_{\text{critical}} = 0.4$, $p_{\text{elderly}} = 0.35$ (from scenario probabilities)
- Since $p_{\text{critical}} < 0.6$ and $p_{\text{elderly}} < 0.4$, constraints are challenging but potentially satisfiable through selective acceptance

## Advanced Extensions

### Multi-Stage Stochastic Programming

Extend to handle multiple stages of uncertainty:
$$\max \mathbb{E}[\text{Stage 1}] + \mathbb{E}[\max \text{Stage 2 given Stage 1}] + ...$$

### Robust Optimization

Handle uncertainty through worst-case scenarios:
$$\max \min_{\text{scenario}} \text{Objective}(\text{scenario})$$

### Chance Constraints

Probabilistic constraint satisfaction:
$$P(\text{Constraint satisfied}) \geq 1 - \alpha$$

## Adaptive Linear Programming with Primal-Dual Pricing

### Motivation

Standard LP formulations face challenges in stochastic environments:
1. **Static Formulation**: Fixed parameters don't adapt to constraint difficulty patterns
2. **Infeasibility Handling**: Basic LP fails when problems become infeasible
3. **No Price Discovery**: Lacks mechanism to understand constraint "shadow prices"

### Primal-Dual Framework

The adaptive approach maintains both primal decisions and dual prices:

**Primal Problem**: Make acceptance decisions $x_t$
**Dual Problem**: Discover shadow prices $\pi_j$ for each constraint $j$

### Shadow Price Dynamics

For each constraint $j$, maintain an exponentially weighted shadow price:
$$\pi_j^{(t+1)} = \alpha \cdot \pi_j^{(\text{current})} + (1-\alpha) \cdot \pi_j^{(t)}$$

Where $\pi_j^{(\text{current})}$ is derived from LP dual variables when feasible, or estimated from constraint violations when infeasible.

### Adaptive Acceptance Threshold

The acceptance threshold evolves based on constraint tightness:
$$\theta^{(t+1)} = \theta^{(t)} + \beta \cdot \left(\mathbb{I}_{\text{LP success}} - \bar{V}^{(t)}\right)$$

Where:
- $\beta$: learning rate
- $\mathbb{I}_{\text{LP success}}$: indicator of LP feasibility
- $\bar{V}^{(t)}$: average constraint violation rate

### Price-Based Decision Making

When LP becomes infeasible, use shadow prices for decision-making:

**Entity Value Calculation**:
$$V(e_t) = \sum_{j} \pi_j \cdot \left(\mathbf{1}_{e_t \text{ has } a_j} - \theta_j\right) \cdot D_j^{(t)}$$

Where $D_j^{(t)}$ is the current deficit for constraint $j$.

**Decision Rule**:
$$x_t = \begin{cases}
\text{LP solution} & \text{if LP feasible} \\
\mathbf{1}_{V(e_t) > \theta^{(t)} \cdot \bar{\pi}} & \text{if LP infeasible}
\end{cases}$$

### Adaptive Horizon Strategy

Dynamically adjust lookahead horizon based on constraint tightness:
$$H^{(t)} = H_{\text{base}} \cdot \left(1 + \gamma \cdot \bar{T}^{(t)}\right)$$

Where:
- $H_{\text{base}}$: base horizon length
- $\gamma$: horizon expansion factor
- $\bar{T}^{(t)}$: average constraint tightness across all constraints

### Constraint Relaxation Mechanism

When constraints consistently cause infeasibility, apply adaptive relaxation:

For constraint $j$ with shadow price $\pi_j > \pi_{\text{threshold}}$:
$$\theta_j^{\text{relaxed}} = \theta_j \cdot (1 - \epsilon \cdot \min(1, \pi_j / \pi_{\text{threshold}}))$$

This temporarily reduces constraint requirements for consistently problematic constraints.

### Complete Adaptive LP Algorithm

```
1. Initialize: π_j = 0, θ = θ_0, H = H_base
2. For each arriving entity e_t:
   a. Update constraint tightness measures
   b. Adapt horizon: H^(t) = f(constraint_tightness)
   c. Formulate LP with current state and adaptive parameters
   d. Attempt LP solution:
      if feasible:
        - Extract dual variables → update π_j
        - Use LP decision x_t*
        - Update θ based on success
      else:
        - Calculate V(e_t) using shadow prices
        - Make price-based decision
        - Update θ based on failure
   e. Update shadow prices and constraint history
```

## Rolling Horizon Linear Programming

### Multi-Stage Decision Framework

Instead of single-entity decisions, optimize over a sliding window of entities:

**Decision Variables**:
$$\mathbf{x}^{(t)} = [x_t, x_{t+1}, \ldots, x_{t+H-1}]$$

**Rolling Objective**:
$$\max \sum_{i=0}^{H-1} R(e_{t+i}, x_{t+i}) \text{ subject to horizon constraints}$$

### Receding Horizon Implementation

At each time step:
1. Solve multi-stage LP over horizon $[t, t+H-1]$
2. Implement only first decision $x_t^*$
3. Shift horizon forward: $[t+1, t+H]$
4. Add new entity $e_{t+H}$ to horizon
5. Repeat optimization

### Horizon Memory Integration

Maintain memory of previous horizon solutions to warm-start optimization:
$$\mathbf{x}_{\text{init}}^{(t+1)} = [\mathbf{x}^{(t)}_{2:H}, x_{\text{predicted}}]$$

This provides better initial solutions and faster convergence.

### Multi-Stage Constraint Propagation

Constraints are enforced across the entire horizon:

**Capacity Propagation**:
$$\sum_{i=0}^{H-1} x_{t+i} \leq N_{\text{remaining}}^{(t)}$$

**Attribute Propagation**:
$$c_j^{(t)} + \sum_{i=0}^{H-1} x_{t+i} \cdot \mathbf{1}_{e_{t+i} \text{ has } a_j} \geq \theta_j \cdot \left(n_t + \sum_{i=0}^{H-1} x_{t+i}\right)$$

### Computational Efficiency

**Warm Starting**: Use previous solution as starting point
**Incremental Updates**: Only recompute changed constraints
**Horizon Pruning**: Limit horizon size based on remaining capacity

## Practical Implementation Guidelines

### Configuration Parameters for Adaptive LP

**Primal-Dual Parameters**:
- `dual_memory_length`: Number of dual solutions to remember (default: 50)
- `price_smoothing_factor`: EMA smoothing for shadow prices (default: 0.3)
- `learning_rate`: Rate for threshold adaptation (default: 0.1)

**Adaptive Parameters**:
- `min_acceptance_threshold`: Lower bound for threshold (default: 0.1)
- `max_acceptance_threshold`: Upper bound for threshold (default: 0.9)
- `constraint_buffer`: Safety margin for constraints (default: 0.05)

**Rolling Horizon Parameters**:
- `adaptive_horizon`: Enable dynamic horizon adjustment (default: true)
- `min_horizon`: Minimum lookahead window (default: 10)
- `max_horizon`: Maximum lookahead window (default: 100)

### When to Use Each Approach

**Standard LP**: 
- Stable, well-understood environments
- Constraints are consistently feasible
- Interpretability is important

**Adaptive LP**:
- Dynamic environments with changing constraint difficulty
- Problems with frequent infeasibility
- Need for online learning and adaptation

**Rolling Horizon LP**:
- Complex multi-stage dependencies
- Long-term planning required
- Computational resources available for larger optimizations

### Performance Considerations

**Adaptive LP Complexity**:
- Time: O(K) per decision for price updates + O(H × K) for LP when feasible
- Space: O(M × K) for dual memory storage

**Rolling Horizon Complexity**:
- Time: O(H² × K) per decision for multi-stage LP
- Space: O(H × K) for horizon state storage

Where H = horizon size, K = number of constraints, M = memory length.

## Software Available
- **LP Solver**: CPLEX, Gurobi, or open-source COIN-OR
- **Python Libraries**: `pulp`, `scipy.optimize`, or `cvxpy`
- **Performance**: Vectorized operations with NumPy

## Conclusion

Linear Programming provides a mathematically rigorous and practical approach to Stochastic Constraint Satisfaction Problems. The evolution from basic LP formulations to adaptive primal-dual approaches demonstrates the flexibility and power of optimization-based methods.

### Method Selection Guidelines

**Basic LP** is suitable when:
- **Interpretability** is important for decision validation
- **Constraint satisfaction** is critical but environment is stable
- **Mathematical rigor** is required for auditing
- **Computational simplicity** is preferred

**Adaptive LP with Primal-Dual Pricing** is useful when:
- **Dynamic environments** with changing constraint difficulty
- **Frequent infeasibility** occurs in basic LP formulations
- **Online learning** and adaptation to problem patterns is needed
- **Robust performance** across varying scenarios is required

**Rolling Horizon LP** is optimal for:
- **Complex multi-stage dependencies** between decisions
- **Long-term planning** with sophisticated temporal constraints
- **High-stakes decisions** where computational cost is acceptable
- **Environments** where lookahead provides significant value

### Integration with Other Approaches

The adaptive LP framework provides a strong foundation for hybrid strategies:
- **LP + Reinforcement Learning**: Use LP as a baseline policy for RL exploration
- **LP + Machine Learning**: Leverage ML to improve probability estimates
- **LP + Robust Optimization**: Combine adaptive pricing with worst-case analysis
- **LP + Simulation**: Use LP solutions to guide Monte Carlo tree search

The mathematical rigor of LP combined with adaptive mechanisms creates a powerful toolkit for stochastic constraint satisfaction that balances optimality, robustness, and computational tractability.

## References

### Linear Programming Foundations
- **Bertsimas, D., & Tsitsiklis, J. N. (1997).** *Introduction to Linear Optimization*. Athena Scientific.
  - *Chapters 1-4*: Fundamental LP theory and algorithms

- **Vanderbei, R. J. (2020).** *Linear Programming: Foundations and Extensions* (5th ed.). Springer.
  - *Chapter 1*: LP formulation techniques
  - *Chapter 4*: Duality theory and sensitivity analysis

### Stochastic Programming
- **Birge, J. R., & Louveaux, F. (2011).** *Introduction to Stochastic Programming* (2nd ed.). Springer.
  - *Chapter 2*: Two-stage stochastic programming with recourse
  - *Chapter 4*: Multistage stochastic programming

### Online and Dynamic Programming
- **Powell, W. B. (2007).** *Approximate Dynamic Programming: Solving the Curses of Dimensionality*. John Wiley & Sons.
  - *Chapter 12*: Linear programming and approximate dynamic programming
