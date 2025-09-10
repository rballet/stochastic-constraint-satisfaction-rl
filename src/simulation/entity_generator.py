"""
Entity generation for stochastic constraint satisfaction problem simulation.
"""

import random
import numpy as np
from typing import Optional, Dict, List
from ..core.types import Entity, Scenario, EntityGenerator


class BasicEntityGenerator:
    """Basic entity generator using independent attribute sampling."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.random_state = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        
    def generate_entity(self, scenario: Scenario) -> Entity:
        """Generate an entity with attributes sampled independently."""
        attributes = {}
        
        for attribute in scenario.attributes:
            prob = scenario.attribute_probabilities.get(attribute, 0.5)
            attributes[attribute] = self.random_state.random() < prob
            
        return Entity(attributes=attributes)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator state."""
        if seed is not None:
            self.seed = seed
        self.random_state = random.Random(self.seed)
        self.np_random = np.random.RandomState(self.seed)
    
    @property
    def name(self) -> str:
        return "BasicEntityGenerator"


class CorrelatedEntityGenerator:
    """Entity generator that handles attribute correlations."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.random_state = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        
    def generate_entity(self, scenario: Scenario) -> Entity:
        """Generate an entity with correlated attributes."""
        attributes = {}
        
        # First, sample attributes in dependency order
        # For simplicity, we'll sample in the order they appear in scenario.attributes
        for attribute in scenario.attributes:
            prob = self._get_conditional_probability(attribute, attributes, scenario)
            attributes[attribute] = self.random_state.random() < prob
            
        return Entity(attributes=attributes)
    
    def _get_conditional_probability(
        self, 
        attribute: str, 
        existing_attributes: Dict[str, bool], 
        scenario: Scenario
    ) -> float:
        """Get conditional probability of attribute given existing attributes."""
        base_prob = scenario.attribute_probabilities.get(attribute, 0.5)
        
        # Apply correlations
        adjusted_prob = base_prob
        for (attr1, attr2), correlation in scenario.attribute_correlations.items():
            if attr2 == attribute and attr1 in existing_attributes:
                # Simple correlation adjustment
                if existing_attributes[attr1]:
                    # Positive correlation increases probability if attr1 is True
                    adjusted_prob += correlation * (1 - base_prob)
                else:
                    # Negative correlation when attr1 is False
                    adjusted_prob -= correlation * base_prob
                    
        return max(0.0, min(1.0, adjusted_prob))
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator state."""
        if seed is not None:
            self.seed = seed
        self.random_state = random.Random(self.seed)
        self.np_random = np.random.RandomState(self.seed)
    
    @property
    def name(self) -> str:
        return "CorrelatedEntityGenerator"


class MultivariateEntityGenerator:
    """Advanced entity generator using multivariate distributions."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self._fitted_models = {}
        
    def fit_scenario(self, scenario: Scenario) -> None:
        """Pre-fit the generator for a specific scenario using a calibrated Gaussian copula.

        We solve for a latent normal correlation matrix that, after thresholding
        at Φ^{-1}(p_i), approximately reproduces the target Bernoulli correlations.
        """
        from scipy.stats import norm, multivariate_normal  # lazy import

        n_attrs = len(scenario.attributes)
        attr_to_idx = {attr: i for i, attr in enumerate(scenario.attributes)}
        probs = [float(scenario.attribute_probabilities.get(attr, 0.5)) for attr in scenario.attributes]
        thresholds = np.array([norm.ppf(p) for p in probs], dtype=float)

        # Helper: binary correlation achieved by latent normal corr r for pair (i, j)
        def bernoulli_corr_from_latent(r: float, pi: float, pj: float, ai: float, aj: float) -> float:
            cov = [[1.0, r], [r, 1.0]]
            p11 = multivariate_normal.cdf([ai, aj], mean=[0.0, 0.0], cov=cov)
            cov_bern = p11 - (pi * pj)
            denom = np.sqrt(pi * (1.0 - pi) * pj * (1.0 - pj))
            if denom <= 1e-12:
                return 0.0
            return float(cov_bern / denom)

        # Calibrate pairwise latent correlations via bisection for each pair with provided target
        latent_corr = np.eye(n_attrs, dtype=float)
        for (attr1, attr2), target in scenario.attribute_correlations.items():
            if (attr1 not in attr_to_idx) or (attr2 not in attr_to_idx):
                continue
            i, j = attr_to_idx[attr1], attr_to_idx[attr2]
            pi, pj = probs[i], probs[j]
            ai, aj = thresholds[i], thresholds[j]
            # Bisection bounds for latent correlation
            lo, hi = -0.999, 0.999
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                corr_mid = bernoulli_corr_from_latent(mid, pi, pj, ai, aj)
                if corr_mid < target:
                    lo = mid
                else:
                    hi = mid
            latent_corr[i, j] = latent_corr[j, i] = 0.5 * (lo + hi)

        # Ensure PSD by clipping eigenvalues
        w, V = np.linalg.eigh(latent_corr)
        w = np.clip(w, 1e-6, None)
        latent_corr_psd = (V @ np.diag(w) @ V.T)
        # Normalize diagonal back to 1
        d = np.sqrt(np.clip(np.diag(latent_corr_psd), 1e-12, None))
        latent_corr_psd = latent_corr_psd / np.outer(d, d)

        self._fitted_models[scenario.name] = {
            'attributes': scenario.attributes,
            'probabilities': scenario.attribute_probabilities,
            'latent_corr': latent_corr_psd,
            'thresholds': thresholds,
            'attr_to_idx': attr_to_idx
        }
    
    def generate_entity(self, scenario: Scenario) -> Entity:
        """Generate an entity using fitted multivariate model."""
        if scenario.name not in self._fitted_models:
            self.fit_scenario(scenario)
        
        model = self._fitted_models[scenario.name]
        n_attrs = len(model['attributes'])
        # Sample latent normal with calibrated correlation
        normal_samples = self.np_random.multivariate_normal(
            mean=np.zeros(n_attrs),
            cov=model['latent_corr']
        )
        # Threshold at quantiles for each attribute to achieve correct marginals
        thresholds = model['thresholds']
        attributes = {}
        for i, attr in enumerate(model['attributes']):
            attributes[attr] = normal_samples[i] <= thresholds[i]
            
        return Entity(attributes=attributes)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator state."""
        if seed is not None:
            self.seed = seed
        self.np_random = np.random.RandomState(self.seed)
    
    @property
    def name(self) -> str:
        return "MultivariateEntityGenerator"


def create_entity_generator(generator_type: str = "basic", seed: Optional[int] = None) -> EntityGenerator:
    """Factory function to create entity generators."""
    generators = {
        "basic": BasicEntityGenerator,
        "correlated": CorrelatedEntityGenerator,
        "multivariate": MultivariateEntityGenerator
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    return generators[generator_type](seed=seed)