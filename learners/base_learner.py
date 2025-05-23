"""
Abstract base learner class for the Bayesian optimization system.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


class BaseLearner(ABC):
    """
    Abstract base class for machine learning models in Bayesian optimization.
    
    This class defines the interface that all learners must implement to be
    compatible with the BayesianHyperparameterOptimizer.
    """
    
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        """
        Initialize the learner with feature specifications.
        
        Args:
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.all_features = numerical_features + categorical_features
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the learner name/identifier."""
        pass
    
    @abstractmethod
    def get_search_space(self) -> List[Dict]:
        """
        Return hyperparameter search space definition for Ax optimization.
        
        Returns:
            List of parameter dictionaries in Ax format with keys:
            - name: parameter name
            - type: "range" or "choice"
            - bounds: [min, max] for range parameters
            - values: list of values for choice parameters
            - value_type: "int" or "float"
        """
        pass
    
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare data for training/prediction (e.g., encoding categorical features).
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (to fit encoders) or validation data
            
        Returns:
            Processed dataframe ready for model training/prediction
        """
        pass
    
    @abstractmethod
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        w_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        w_val: pd.Series,
        params: Dict[str, Any], 
        max_iterations: int
    ) -> Any:
        """
        Train a single model with given parameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            w_train: Training weights
            X_val: Validation features
            y_val: Validation targets
            w_val: Validation weights
            params: Hyperparameters to use for training
            max_iterations: Maximum number of training iterations
            
        Returns:
            Trained model object
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: pd.DataFrame, iteration: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            model: Trained model object
            X: Features for prediction
            iteration: Specific iteration to use (for boosting models with early stopping)
            
        Returns:
            Prediction array
        """
        pass
    
    def supports_iteration_prediction(self) -> bool:
        """
        Whether this learner supports prediction at specific iterations.
        
        This is typically True for boosting models (LightGBM, CatBoost, XGBoost)
        and False for models like Random Forest, Neural Networks, etc.
        
        Returns:
            True if the learner supports iteration-specific predictions
        """
        return False
    
    def get_default_max_iterations(self) -> int:
        """
        Get the default maximum iterations for this learner.
        
        Returns:
            Default max iterations (can be overridden by optimizer config)
        """
        return 100