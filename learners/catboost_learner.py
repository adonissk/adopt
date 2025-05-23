"""
CatBoost learner implementation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from .base_learner import BaseLearner

try:
    import catboost as cb
except ImportError:
    cb = None


class CatBoostLearner(BaseLearner):
    """
    CatBoost learner implementation for Bayesian optimization.
    
    Handles CatBoost-specific training, prediction, and hyperparameter search space.
    CatBoost handles categorical features natively without encoding.
    """
    
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        if cb is None:
            raise ImportError("catboost package is required for CatBoostLearner. Install with: pip install catboost")
        
        super().__init__(numerical_features, categorical_features)
    
    @property
    def name(self) -> str:
        return "catboost"
    
    def supports_iteration_prediction(self) -> bool:
        return True
    
    def get_search_space(self) -> List[Dict]:
        """Create CatBoost hyperparameter search space."""
        return [
            {
                "name": "depth",
                "type": "range",
                "bounds": [4, 10],
                "value_type": "int"
            },
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.01, 0.3],
                "value_type": "float"
            },
            {
                "name": "l2_leaf_reg",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "float"
            },
            {
                "name": "border_count",
                "type": "range",
                "bounds": [32, 255],
                "value_type": "int"
            },
            {
                "name": "rsm",
                "type": "range",
                "bounds": [0.5, 1.0],
                "value_type": "float"
            },
            {
                "name": "subsample",
                "type": "range",
                "bounds": [0.5, 1.0],
                "value_type": "float"
            },
            {
                "name": "bagging_temperature",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float"
            },
            {
                "name": "min_data_in_leaf",
                "type": "range",
                "bounds": [1, 100],
                "value_type": "int"
            },
            {
                "name": "one_hot_max_size",
                "type": "range",
                "bounds": [2, 255],
                "value_type": "int"
            },
            {
                "name": "leaf_estimation_iterations",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int"
            }
        ]
    
    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare data for CatBoost training.
        
        CatBoost handles categorical features natively, so minimal preprocessing is needed.
        Just ensure categorical features are kept as strings/objects.
        """
        df_processed = df.copy()
        
        # Ensure categorical features are of object type for CatBoost native support
        for cat_col in self.categorical_features:
            if df_processed[cat_col].dtype != 'object':
                df_processed[cat_col] = df_processed[cat_col].astype(str)
        
        return df_processed
    
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
    ) -> cb.CatBoostRegressor:
        """Train a CatBoost model with given parameters."""
        
        # Create CatBoost datasets with native categorical support
        train_pool = cb.Pool(
            X_train, 
            label=y_train, 
            weight=w_train,
            cat_features=self.categorical_features
        )
        val_pool = cb.Pool(
            X_val, 
            label=y_val, 
            weight=w_val,
            cat_features=self.categorical_features
        )
        
        # CatBoost parameters
        cb_params = {
            "iterations": max_iterations,
            "objective": "RMSE",
            "verbose": False,
            "random_seed": 42,
            **params
        }
        
        # Train model
        model = cb.CatBoostRegressor(**cb_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False,
            use_best_model=False  # We handle iteration selection manually
        )
        
        return model
    
    def predict(self, model: cb.CatBoostRegressor, X: pd.DataFrame, iteration: Optional[int] = None) -> np.ndarray:
        """Make predictions using the trained CatBoost model."""
        if iteration is not None:
            # Predict with specific iteration (CatBoost uses ntree_end)
            return model.predict(X, ntree_end=iteration)
        else:
            # Predict with all iterations
            return model.predict(X)