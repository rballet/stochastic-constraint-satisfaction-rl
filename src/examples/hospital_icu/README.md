# Hospital ICU Admission Problem

## Problem Description

The Hospital ICU Admission Problem is a stochastic constraint satisfaction problem where hospital administrators must make real-time decisions about ICU bed allocation for incoming patients. This scenario involves:

- **Sequential Decision-Making**: Patients arrive one by one, requiring immediate admission decisions
- **Limited Capacity**: Only 50 ICU beds available  
- **Medical Constraints**: Must maintain specific patient mix ratios for optimal care
- **Uncertainty**: Patient characteristics are only known upon arrival
- **Quality Optimization**: Maximize overall patient care quality while meeting constraints

## Problem Setup

### Patient Attributes
Each arriving patient has the following binary attributes:
- **`critical_condition`**: Patient requires immediate life support (probability: 0.4)
- **`elderly`**: Patient is over 65 years old (probability: 0.35)
- **`has_insurance`**: Patient has health insurance coverage (probability: 0.85)
- **`high_risk`**: Patient has high-risk comorbidities (probability: 0.3)
- **`emergency_case`**: Patient arrived via emergency transport (probability: 0.25)

### Medical Constraints
The ICU must maintain the following patient mix ratios:
1. **Critical Care Requirement**: At least 60% of admitted patients must be in critical condition
2. **Elderly Care Capacity**: At least 40% of admitted patients should be elderly (specialized geriatric care)
3. **High-Risk Management**: At least 25% of admitted patients should be high-risk cases

### Termination Conditions
The simulation ends when:
- **Capacity Reached**: All 50 ICU beds are occupied, OR
- **Rejection Limit**: 200 patients have been rejected (indicating poor resource utilization)

### Success Criteria
A successful allocation strategy must:
- Fill the ICU to capacity (50 patients)
- Satisfy all medical constraints
- Minimize patient rejections (optimize care access)

## Scenarios

### Scenario 1: Standard Operating Conditions
- Standard patient arrival distributions
- All constraints equally weighted
- Baseline scenario for strategy comparison

### Scenario 2: High-Acuity Period  
- Increased critical patient arrivals (probability: 0.6)
- Higher elderly patient ratio (probability: 0.45)
- Simulates flu season or health crisis conditions

### Scenario 3: Emergency Surge
- Increased emergency cases (probability: 0.4)
- Higher high-risk patient ratio (probability: 0.45)
- Reduced insurance coverage (probability: 0.7)
- Simulates disaster or epidemic conditions

## Ethical Considerations

This problem formulation focuses on:
- **Medical Necessity**: Prioritizing patients based on clinical needs
- **Resource Optimization**: Maximizing beneficial use of limited ICU capacity
- **Regulatory Compliance**: Meeting medical guidelines and standards
- **Equitable Access**: Ensuring fair allocation procedures

Note: This is a simplified model for research purposes. Real medical triage involves many additional factors including specific medical conditions, prognosis, treatment requirements, and ethical guidelines that are not captured in this abstraction.

## Implementation Files

- `scenarios.py`: Defines the three scenarios with different patient distributions
- `patient_generator.py`: Generates patients with correlated medical attributes
- `icu_strategies.py`: Basic allocation strategies (random, greedy, priority-based)
- `icu_env.py`: Gymnasium environment for RL training
- `run_icu_simulation.py`: Script to run and evaluate strategies