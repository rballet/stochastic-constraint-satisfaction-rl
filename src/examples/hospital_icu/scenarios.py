"""
ICU admission scenarios with different patient arrival patterns.
"""

from ...core.types import Scenario, Constraint


# Constants for scenarios
ICU_CAPACITY = 100
ICU_MAX_REJECTIONS = 1000


def create_icu_scenario_1() -> Scenario:
    """
    Scenario 1: Standard Operating Conditions
    
    Standard patient arrival distributions with all constraints equally weighted.
    Baseline scenario for strategy comparison.
    """
    return Scenario(
        name="icu_standard",
        description="Standard ICU operating conditions",
        attributes=["critical_condition", "elderly", "has_insurance", "high_risk", "emergency_case"],
        constraints=[
            Constraint(
                attribute="critical_condition", 
                min_percentage=0.60, 
                description="At least 60% critical patients"
            ),
            Constraint(
                attribute="elderly", 
                min_percentage=0.40, 
                description="At least 40% elderly patients"
            ),
            Constraint(
                attribute="high_risk", 
                min_percentage=0.25, 
                description="At least 25% high-risk patients"
            ),
        ],
        attribute_probabilities={
            "critical_condition": 0.40,
            "elderly": 0.35,
            "has_insurance": 0.85,
            "high_risk": 0.30,
            "emergency_case": 0.25,
        },
        attribute_correlations={
            ("elderly", "high_risk"): 0.3,  # Elderly patients more likely to be high-risk
            ("critical_condition", "emergency_case"): 0.4,  # Critical patients often arrive via emergency
            ("high_risk", "critical_condition"): 0.25,  # High-risk patients more likely to be critical
            ("elderly", "has_insurance"): 0.2,  # Elderly patients more likely to have insurance
        }
    )


def create_icu_scenario_2() -> Scenario:
    """
    Scenario 2: High-Acuity Period
    
    Increased critical patient arrivals and higher elderly ratio.
    Simulates flu season or health crisis conditions.
    """
    return Scenario(
        name="icu_high_acuity",
        description="High-acuity period (flu season/health crisis)",
        attributes=["critical_condition", "elderly", "has_insurance", "high_risk", "emergency_case"],
        constraints=[
            Constraint(
                attribute="critical_condition", 
                min_percentage=0.60, 
                description="At least 60% critical patients"
            ),
            Constraint(
                attribute="elderly", 
                min_percentage=0.40, 
                description="At least 40% elderly patients"
            ),
            Constraint(
                attribute="high_risk", 
                min_percentage=0.25, 
                description="At least 25% high-risk patients"
            ),
        ],
        attribute_probabilities={
            "critical_condition": 0.60,  # Increased critical cases
            "elderly": 0.45,  # More elderly patients
            "has_insurance": 0.85,
            "high_risk": 0.35,  # Slightly more high-risk
            "emergency_case": 0.30,  # More emergency arrivals
        },
        attribute_correlations={
            ("elderly", "high_risk"): 0.4,  # Stronger correlation during crisis
            ("critical_condition", "emergency_case"): 0.5,
            ("high_risk", "critical_condition"): 0.35,
            ("elderly", "has_insurance"): 0.2,
            ("elderly", "critical_condition"): 0.3,  # Elderly more likely to be critical during crisis
        }
    )


def create_icu_scenario_3() -> Scenario:
    """
    Scenario 3: Emergency Surge
    
    High emergency cases and reduced insurance coverage.
    Simulates disaster or epidemic conditions.
    """
    return Scenario(
        name="icu_emergency_surge",
        description="Emergency surge conditions (disaster/epidemic)",
        attributes=["critical_condition", "elderly", "has_insurance", "high_risk", "emergency_case"],
        constraints=[
            Constraint(
                attribute="critical_condition", 
                min_percentage=0.60, 
                description="At least 60% critical patients"
            ),
            Constraint(
                attribute="elderly", 
                min_percentage=0.40, 
                description="At least 40% elderly patients"
            ),
            Constraint(
                attribute="high_risk", 
                min_percentage=0.25, 
                description="At least 25% high-risk patients"
            ),
        ],
        attribute_probabilities={
            "critical_condition": 0.55,
            "elderly": 0.35,
            "has_insurance": 0.70,  # Reduced insurance coverage
            "high_risk": 0.45,  # Many high-risk patients
            "emergency_case": 0.40,  # High emergency arrivals
        },
        attribute_correlations={
            ("elderly", "high_risk"): 0.35,
            ("critical_condition", "emergency_case"): 0.6,  # Most critical patients arrive via emergency
            ("high_risk", "critical_condition"): 0.4,  # Strong correlation
            ("elderly", "has_insurance"): 0.15,  # Weaker during emergencies
            ("emergency_case", "high_risk"): 0.3,  # Emergency cases often high-risk
            ("has_insurance", "critical_condition"): -0.1,  # Slight negative correlation during crisis
        }
    )


def get_all_icu_scenarios():
    """Get all ICU scenarios for comparison studies."""
    return [
        create_icu_scenario_1(),
        create_icu_scenario_2(),
        create_icu_scenario_3(),
    ]
