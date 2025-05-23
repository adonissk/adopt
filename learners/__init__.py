"""
Modular learner system for Bayesian optimization.

This package provides a plugin-like architecture for adding new machine learning models
to the Bayesian hyperparameter optimization pipeline.
"""

from .factory import LearnerFactory
from .base_learner import BaseLearner

# Import registry to auto-register default learners
from . import registry

__all__ = ['LearnerFactory', 'BaseLearner']