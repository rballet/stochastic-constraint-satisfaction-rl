"""
ICU admission scenarios with different patient arrival patterns.
"""

from ...core.types import Scenario, Constraint


# Constants for scenarios
ICU_CAPACITY = 1000
ICU_MAX_REJECTIONS = 20000


def create_icu_scenario_1() -> Scenario:
    """
    Scenario 1: Standard Operating Conditions
    
    Standard patient arrival distributions with all constraints equally weighted.
    Baseline scenario for strategy comparison.
    """
    return Scenario(
        name="icu_standard",
        category="ICU",
        description="Standard ICU operating conditions",
        attributes=["critical_condition", "elderly", "has_insurance", "high_risk", "emergency_case"],
        capacity=ICU_CAPACITY,
        max_rejections=ICU_MAX_REJECTIONS,
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
        category="ICU",
        description="High-acuity period (flu season/health crisis)",
        attributes=["critical_condition", "elderly", "has_insurance", "high_risk", "emergency_case"],
        capacity=ICU_CAPACITY,
        max_rejections=ICU_MAX_REJECTIONS,
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
        category="ICU",
        description="Emergency surge conditions (disaster/epidemic)",
        attributes=["critical_condition", "elderly", "has_insurance", "high_risk", "emergency_case"],
        capacity=ICU_CAPACITY,
        max_rejections=ICU_MAX_REJECTIONS,
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


def create_icu_scenario_4() -> Scenario:
    """
    Scenario 4: Challenging Negative Correlations
    
    Highly challenging scenario with strong negative correlations between attributes.
    Based on complex patient demographics where attributes compete against each other.
    This scenario tests strategies under adversarial correlation patterns.
    
    Note: The negative correlations represent a challenging theoretical scenario
    where the constraint requirements conflict with natural patient demographics.
    """
    return Scenario(
        name="icu_negative_correlations",
        category="ICU",
        description="Challenging scenario with strong negative correlations",
        attributes=["critical_condition", "elderly", "has_insurance", "high_risk", "emergency_case"],
        capacity=ICU_CAPACITY,
        max_rejections=ICU_MAX_REJECTIONS,
        constraints=[
            Constraint(
                attribute="critical_condition", 
                min_percentage=0.65,  # 65% critical - higher than baseline frequencies
                description="At least 65% critical patients"
            ),
            Constraint(
                attribute="elderly", 
                min_percentage=0.45,  # 45% elderly - challenging given negative correlation with critical
                description="At least 45% elderly patients"
            ),
            Constraint(
                attribute="has_insurance", 
                min_percentage=0.30,  # 30% insured - much lower than typical, but achievable
                description="At least 30% insured patients"
            ),
            Constraint(
                attribute="high_risk", 
                min_percentage=0.75,  # 75% high-risk - very challenging given low base frequency
                description="At least 75% high-risk patients"
            ),
        ],
        attribute_probabilities={
            # Based on relativeFrequencies from the example
            "critical_condition": 0.6265,
            "elderly": 0.47,             
            "has_insurance": 0.06227,    
            "high_risk": 0.398,          
            "emergency_case": 0.35,      
        },
        attribute_correlations={
            # Strong negative correlations that make constraint satisfaction very difficult
            # Note: These represent a challenging theoretical scenario where demographics conflict
            ("critical_condition", "elderly"): -0.469616933267432,        # Strong negative - theoretical challenge
            ("critical_condition", "high_risk"): -0.654940381560618,      # Very strong negative - theoretical challenge
            ("elderly", "has_insurance"): 0.141972591404715,              # Weak positive
            ("elderly", "high_risk"): 0.572406780843645,                  # Strong positive
            ("has_insurance", "high_risk"): 0.144464595056508,            # Weak positive
            ("critical_condition", "has_insurance"): 0.0946331703989159,  # Very weak positive
            
            # Additional challenging correlations
            ("emergency_case", "critical_condition"): -0.1,               
            ("emergency_case", "elderly"): 0.1,                         
            ("emergency_case", "has_insurance"): -0.2,                    
        }
    )


def get_all_icu_scenarios():
    """Get all ICU scenarios for comparison studies."""
    return [
        create_icu_scenario_1(),
        create_icu_scenario_2(),
        create_icu_scenario_3(),
        create_icu_scenario_4(),
    ]
