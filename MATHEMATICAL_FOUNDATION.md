# Mathematical Foundation of Stochastic Constraint Satisfaction Problems

## Overview

Stochastic Constraint Satisfaction Problems (SCSPs) represent a class of optimization problems where decisions must be made sequentially under uncertainty while satisfying multiple constraints simultaneously. Unlike traditional constraint satisfaction problems where all information is known a priori, SCSPs involve probabilistic entity arrivals and incomplete future information.

## Problem Formulation

### Core Mathematical Definition

A Stochastic Constraint Satisfaction Problem is defined as a tuple:

$$\mathcal{S} = \langle \mathcal{A}, \mathcal{C}, \mathcal{D}, \mathcal{P}, \mathcal{T}, \mathcal{R} \rangle$$

Where:
- $\mathcal{A}$ = Set of attributes/features
- $\mathcal{C}$ = Set of constraints
- $\mathcal{D}$ = Decision space (typically $\{0, 1\}$ for accept/reject)
- $\mathcal{P}$ = Probability distributions over attribute combinations
- $\mathcal{T}$ = Termination conditions
- $\mathcal{R}$ = Reward/utility function

### Entity Model

Each arriving entity $e_t$ at time $t$ is characterized by:

$$e_t = (a_1^{(t)}, a_2^{(t)}, \ldots, a_{|\mathcal{A}|}^{(t)})$$

where $a_i^{(t)} \in \{0, 1\}$ indicates the presence/absence of attribute $i$.

**Practical Example (Hospital ICU):**
```
e_t = (critical_condition=1, elderly=0, has_insurance=1, high_risk=1, emergency=0)
```

### State Space

The problem state at time $t$ is defined as:

$$s_t = (n_t, \mathbf{c}_t, r_t)$$

Where:
- $n_t$ = number of entities accepted so far
- $\mathbf{c}_t = (c_1^{(t)}, c_2^{(t)}, \ldots, c_{|\mathcal{A}|}^{(t)})$ = count vector of accepted entities per attribute
- $r_t$ = number of entities rejected so far

**State Evolution:**
$$s_{t+1} = \begin{cases}
(n_t + 1, \mathbf{c}_t + \mathbf{e}_t, r_t) & \text{if } d_t = 1 \text{ (accept)} \\
(n_t, \mathbf{c}_t, r_t + 1) & \text{if } d_t = 0 \text{ (reject)}
\end{cases}$$

## Constraint Framework

### Constraint Definition

Each constraint $\gamma_j \in \mathcal{C}$ is defined as:

$$\gamma_j: \text{attribute}_j \geq \theta_j \cdot n_t$$

Where $\theta_j \in [0, 1]$ is the minimum required proportion for attribute $j$.

**Mathematical Expression:**
$$\frac{c_j^{(t)}}{n_t} \geq \theta_j \quad \forall j \in \mathcal{C}, \text{ when } n_t > 0$$

**ICU Example:**
- Critical care constraint: $\frac{\text{critical patients}}{\text{total patients}} \geq 0.60$
- Elderly care constraint: $\frac{\text{elderly patients}}{\text{total patients}} \geq 0.40$

### Constraint Satisfaction Function

Define the constraint satisfaction indicator:

$$\xi_j(s_t) = \begin{cases}
1 & \text{if } \frac{c_j^{(t)}}{n_t} \geq \theta_j \\
0 & \text{otherwise}
\end{cases}$$

Global constraint satisfaction:
$$\Xi(s_t) = \prod_{j \in \mathcal{C}} \xi_j(s_t)$$

## Stochastic Elements

### Attribute Probability Model

Entities arrive according to a probability distribution over attribute combinations:

$$P(e_t = \mathbf{a}) = P(a_1, a_2, \ldots, a_{|\mathcal{A}|})$$

**Independence Assumption:**
$$P(\mathbf{a}) = \prod_{i=1}^{|\mathcal{A}|} p_i^{a_i} (1-p_i)^{1-a_i}$$

**Correlation Model (More Realistic):**
$$P(\mathbf{a}) = \text{MultivariateBernoulli}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

Where $\boldsymbol{\mu}$ is the mean vector and $\boldsymbol{\Sigma}$ captures correlations.

**ICU Example:**
- $P(\text{critical} = 1) = 0.4$
- $P(\text{elderly} = 1) = 0.35$
- $P(\text{critical} = 1 | \text{elderly} = 1) = 0.6$ (correlation)

### Arrival Process

Entities arrive according to a stochastic process. In the basic formulation:
- **Deterministic arrivals**: One entity per time step
- **Poisson arrivals**: $N_t \sim \text{Poisson}(\lambda)$ entities in interval $[t, t+1)$
- **Batch arrivals**: Fixed batch sizes with random intervals

## Decision Framework

The decision framework captures how we make choices when entities arrive. It mmust balance immediate gains against future opportunities.

### Decision Policy - The Strategy

A **policy** $\pi$ is simply a rule that tells us what to do when an entity arrives:
$$\pi: \mathcal{S} \times \mathcal{E} \rightarrow \mathcal{D}$$


**Real-world examples:**
- **Hospital**: "Accept critical patients if we have beds AND we're not below elderly quota"
- **University**: "Accept high-GPA students unless we need more diversity candidates"
- **Investment**: "Buy tech stocks only if portfolio isn't already 40% tech"

### Finding the Best Strategy

We want the strategy that maximizes long-term value:
$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=1}^T R(s_t, e_t, d_t)\right]$$

**Translation:** Find the decision rule that gives the highest expected total reward over time.

This is fundamentally different from offline optimization because:
1. We don't know what's coming next
2. Today's decision affects tomorrow's opportunities (Vanilla greedy solutions alone do not solve the problem optimally)
3. We must satisfy constraints with uncertain future arrivals

### Value Function - Looking Ahead

The value function needs to answer: "What's the expected future value if I'm in this state and this entity arrives?"

$$V^\pi(s, e) = \mathbb{E}_\pi\left[\sum_{\tau=t}^T R(s_\tau, e_\tau, d_\tau) \mid s_t = s, e_t = e\right]$$

**Bellman Equation** (the core of dynamic programming):
$$V^\pi(s, e) = \max_{d \in \{0,1\}} \left[R(s, e, d) + \mathbb{E}_{e'}[V^\pi(s', e')]\right]$$

**Intuitive meaning:** 
- Take the immediate reward from your decision
- Add the expected value of whatever happens next
- Choose the decision (e.g. accept/reject) that maximizes this total

**Hospital example:** "If I accept this elderly patient now, I get immediate utility but reduce my flexibility for critical patients later. What's the best choice?"

## Utility and Reward Structure

The reward structure is how we "score" our decisions - it encodes what we care about and guides the optimization process.

### Immediate Reward Function - Scoring Each Decision

$$R(s_t, e_t, d_t) = \begin{cases}
R_{\text{accept}}(s_t, e_t) & \text{if } d_t = 1 \\
R_{\text{reject}}(s_t, e_t) & \text{if } d_t = 0
\end{cases}$$

**Design principle:** Rewards should incentivize behavior that leads to overall success (capacity + constraints).

### Common Reward Components

**1. Capacity Utilization** - "Fill the space"
$$\alpha \cdot \frac{n_t}{N_{\max}}$$
- **Intuition:** We get points for using available capacity
- **Example:** Hospital gets $0.1 points per bed occupied out of 50 total

**2. Constraint Penalty** - "Stay on track"
$$-\beta \sum_j \max(0, \theta_j - \frac{c_j^{(t)}}{n_t})$$
- **Intuition:** We lose points when falling behind on constraints
- **Example:** Lose 2 points for every 1% below the 60% critical patient requirement

**3. Terminal Bonus** - "Stick the landing"
$$\gamma \cdot \Xi(s_T) \cdot \mathbf{1}_{n_T = N_{\max}}$$
- **Intuition:** Big bonus for achieving both full capacity AND all constraints
- **Example:** 50-point bonus for filling all beds while meeting medical requirements

**Real-world translation:**
- **Positive rewards:** Successful patient treatment, portfolio returns, student graduation
- **Penalties:** Constraint violations, inefficient resource use, missed opportunities
- **Bonuses:** Completing goals, quality metrics, performance targets

### Multi-Objective Formulation - Balancing Competing Goals

Real systems rarely have single objectives:

$$\max_\pi \sum_{k=1}^K w_k \mathbb{E}[U_k^\pi]$$

**Common objectives in practice:**
- $U_1$: **Efficiency** (capacity utilization)
- $U_2$: **Compliance** (constraint satisfaction)  
- $U_3$: **Equity** (fair treatment across groups)
- $U_4$: **Quality** (outcome measures)

**Weight interpretation:**
- $w_1 = 0.4$: Efficiency is 40% of total score
- $w_2 = 0.3$: Compliance is 30% of total score
- $w_3 = 0.2$: Equity is 20% of total score
- $w_4 = 0.1$: Quality is 10% of total score

**Hospital example:** Balance patient care quality (save lives) vs. resource efficiency (use beds) vs. fairness (serve all populations) vs. compliance (meet medical guidelines).

The choice of reward weights fundamentally shapes system behavior - this is where domain expertise becomes crucial.

## Termination Conditions

Termination conditions define when the decision process ends. Think of these as the "stop signs" that determine when we evaluate final performance.

### When to Stop - The Four Endings

**1. Capacity Reached** - "Mission Accomplished"
$$n_t = N_{\max}$$
- **Meaning:** We've filled all available slots
- **Example:** All 50 ICU beds are occupied
- **Outcome:** Success if constraints are also met

**2. Rejection Limit** - "Too Many Missed Opportunities"  
$$r_t = R_{\max}$$
- **Meaning:** We've rejected too many entities
- **Example:** Turned away 200 patients
- **Interpretation:** Inefficient use of resources; strategy is too picky

**3. Time Limit**
$$t = T_{\max}$$
- **Meaning:** Fixed time horizon reached
- **Example:** End of academic year, fiscal quarter, etc.
- **Outcome:** Evaluate performance based on final state

**4. Infeasibility Detected**
- **Meaning:** Mathematical impossibility to satisfy constraints
- **Example:** Need 60% critical patients but only 10% remain in population
- **Outcome:** Early termination with failure

### Success vs. Failure

**Success Definition:**
$$\text{Success} = \Xi(s_T) \wedge (n_T = N_{\max})$$

**Translation:** Success = (All constraints satisfied) AND (Full capacity reached)

**Performance Spectrum:**
- **Perfect Success:** Full capacity + all constraints met
- **Partial Success:** High capacity but some constraint violations  
- **Efficiency Failure:** Constraints met but low capacity utilization
- **Complete Failure:** Neither capacity nor constraints achieved

**Real-world implications:**
- **Hospital:** Success = full ICU with proper patient mix
- **University:** Success = full enrollment with diversity targets
- **Investment:** Success = full allocation with risk constraints

**Strategic considerations:**
- **Conservative approach:** High constraint satisfaction, risk of low capacity
- **Aggressive approach:** High capacity utilization, risk of constraint violations
- **Balanced approach:** Optimize for both objectives simultaneously

## Complexity Analysis

### Decision Complexity

**State Space Size:** $|\mathcal{S}| = O(N_{\max}^{|\mathcal{A}|+1} \cdot R_{\max})$

**Action Space:** $|\mathcal{D}| = 2$ (binary decisions)

**Horizon:** $T \leq N_{\max} + R_{\max}$ (maximum possible steps)

### Computational Challenges

1. **Curse of Dimensionality**: State space grows exponentially with attributes
2. **Partial Observability**: Future entity distributions may be unknown
3. **Multi-Constraint Coordination**: Satisfying multiple constraints simultaneously
4. **Real-time Decisions**: Limited computation time per decision

## Key Mathematical Properties

### Monotonicity Properties

**Constraint Satisfaction Monotonicity:**
If $\frac{c_j^{(t)}}{n_t} < \theta_j$, then accepting entities with attribute $j$ improves constraint satisfaction.

**Capacity Monotonicity:**
Accepting any entity increases capacity utilization: $n_{t+1} \geq n_t$.

### Feasibility Conditions

**Necessary Condition for Feasibility:**
$$\forall j \in \mathcal{C}: \theta_j \leq p_j \cdot \frac{N_{\max}}{N_{\max} + R_{\max}}$$

Where $p_j$ is the marginal probability of attribute $j$.

**Sufficient Condition:**
If all constraints can be satisfied independently with positive margin under the given arrival distribution.

### Regret Bounds

For online algorithms, define regret as:
$$\text{Regret}_T = V^*(s_0, e_0) - \mathbb{E}[V^\pi(s_0, e_0)]$$

Where $V^*$ is the optimal offline solution with complete information.

## Examples and Intuition

### Example 1: Simple Resource Allocation

**Setup:**
- 2 attributes: {priority, regular}
- 1 constraint: ≥50% priority items
- Capacity: 10 items
- Arrival probabilities: P(priority) = 0.3

**Mathematical Model:**
- State: $(n, c_{\text{priority}}, r)$
- Constraint: $\frac{c_{\text{priority}}}{n} \geq 0.5$ when $n > 0$
- Decision challenge: Accept regular items early or wait for priority items?

### Example 2: Multi-Constraint Portfolio

**Setup:**
- 3 attributes: {tech, healthcare, finance}
- 2 constraints: ≥30% tech, ≥20% healthcare
- Complex correlations between sectors

The stochastic element makes this fundamentally different from deterministic portfolio optimization - decisions must be made without knowing future opportunities.

## Practical Implications

1. **Risk-Reward Tradeoff**: Conservative strategies may fail to reach capacity; aggressive strategies may violate constraints

2. **Information Value**: Knowing arrival distributions improves performance, but perfect information is rarely available

3. **Adaptability**: Effective strategies must adapt to changing constraint satisfaction status

4. **Fairness Considerations**: Mathematical formulation must encode equity and fairness requirements explicitly

## Related Problems

**Relationship to Classical Problems:**
- **Online Algorithms**: Similar to online packing/matching with constraints
- **Stochastic Programming**: Two-stage stochastic programs with recourse
- **Markov Decision Processes**: Sequential decision-making under uncertainty
- **Multi-Armed Bandits**: Exploration-exploitation with constraint satisfaction

This mathematical foundation provides the theoretical basis for developing and analyzing solution approaches including linear programming, dynamic programming, reinforcement learning, and heuristic methods.

## References

### Foundational Theory
- **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  - *Chapters 3-4*: Markov Decision Processes and Dynamic Programming foundations
  - *Chapter 6*: Temporal-Difference Learning for online decision making

- **Puterman, M. L. (2014).** *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. John Wiley & Sons.
  - *Part I*: Mathematical foundations of sequential decision problems under uncertainty

### Stochastic Programming and Online Algorithms
- **Birge, J. R., & Louveaux, F. (2011).** *Introduction to Stochastic Programming* (2nd ed.). Springer.
  - *Chapter 1*: Modeling with recourse for decision making under uncertainty

### Constraint Satisfaction and Multi-Objective Optimization
- **Rossi, F., Van Beek, P., & Walsh, T. (Eds.). (2006).** *Handbook of Constraint Programming*. Elsevier.
  - *Chapter 2*: Constraint satisfaction problems - foundational concepts

- **Miettinen, K. (2012).** *Nonlinear Multiobjective Optimization*. Springer Science & Business Media.
  - *Chapter 2*: Mathematical foundations of multi-objective optimization

### Algorithmic Approaches
- **Powell, W. B. (2007).** *Approximate Dynamic Programming: Solving the Curses of Dimensionality*. John Wiley & Sons.
  - *Chapters 3-5*: Value function approximation for large-scale sequential decision problems

- **Karp, R. M., Vazirani, U. V., & Vazirani, V. V. (1990).** "An optimal algorithm for on-line bipartite matching." *Proceedings of the twenty-second annual ACM symposium on Theory of computing*, 352-358.
