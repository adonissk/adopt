"""
Factory for creating and managing learner instances.
"""

from typing import Dict, List, Type
from .base_learner import BaseLearner


class LearnerFactory:
    """
    Factory for creating learner instances.
    
    Provides a registry system for learner types, allowing easy extension
    of the optimization system with new models.
    """
    
    _learners: Dict[str, Type[BaseLearner]] = {}
    
    @classmethod
    def register_learner(cls, name: str, learner_class: Type[BaseLearner]):
        """
        Register a new learner type.
        
        Args:
            name: Unique identifier for the learner
            learner_class: Class that implements BaseLearner interface
        """
        if not issubclass(learner_class, BaseLearner):
            raise ValueError(f"Learner class must inherit from BaseLearner, got {learner_class}")
        
        cls._learners[name] = learner_class
        print(f"Registered learner: {name}")
    
    @classmethod
    def create_learner(
        cls, 
        name: str, 
        numerical_features: List[str], 
        categorical_features: List[str]
    ) -> BaseLearner:
        """
        Create a learner instance.
        
        Args:
            name: Learner identifier
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            
        Returns:
            Learner instance
            
        Raises:
            ValueError: If learner name is not registered
        """
        if name not in cls._learners:
            available = list(cls._learners.keys())
            raise ValueError(f"Unknown learner: {name}. Available learners: {available}")
        
        learner_class = cls._learners[name]
        return learner_class(numerical_features, categorical_features)
    
    @classmethod
    def get_available_learners(cls) -> List[str]:
        """
        Get list of available learner names.
        
        Returns:
            List of registered learner identifiers
        """
        return list(cls._learners.keys())
    
    @classmethod
    def is_learner_available(cls, name: str) -> bool:
        """
        Check if a learner is available.
        
        Args:
            name: Learner identifier
            
        Returns:
            True if learner is registered, False otherwise
        """
        return name in cls._learners
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered learners (primarily for testing)."""
        cls._learners.clear()