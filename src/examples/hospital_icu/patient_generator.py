"""
Patient generation for ICU admission simulation using the generic entity generator.
"""

from typing import Optional
from ...simulation.entity_generator import create_entity_generator, MultivariateEntityGenerator


def create_icu_patient_generator(generator_type: str = "multivariate", seed: Optional[int] = None):
    """
    Create a patient generator for ICU scenarios.
    
    Uses the generic entity generator framework with multivariate generation
    to accurately capture complex medical attribute correlations.
    
    Args:
        generator_type: Type of generator ("basic", "correlated", "multivariate")  
        seed: Random seed for reproducibility
        
    Returns:
        Entity generator suitable for ICU patient generation
    """
    return create_entity_generator(generator_type, seed)


# Alias for convenience
ICUPatientGenerator = MultivariateEntityGenerator