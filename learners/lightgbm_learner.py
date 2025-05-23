"""
LightGBM learner implementation.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder

from .base_learner import BaseLearner


class LightGBMLearner(BaseLearner):
    """
    LightGBM learner implementation for Bayesian optimization.
    
    Handles LightGBM-specific training, prediction, and hyperparameter search space.
    """
    
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        super().__init__(numerical_features, categorical_features)
        self.categorical_encoders = {}
    
    @property
    def name(self) -> str:
        return "lightgbm"
    
    def supports_iteration_prediction(self) -> bool:
        return True
    
    def get_search_space(self) -> List[Dict]:
        """Create LightGBM hyperparameter search space."""
        return [
            {
                "name": "num_leaves",
                "type": "range",
                "bounds": [10, 300],
                "value_type": "int"
            },
            {
                "name": "learning_rate",
                "type": "range", 
                "bounds": [0.01, 0.3],
                "value_type": "float"
            },
            {
                "name": "feature_fraction",
                "type": "range",
                "bounds": [0.4, 1.0],
                "value_type": "float"
            },
            {
                "name": "bagging_fraction",
                "type": "range",
                "bounds": [0.4, 1.0],
                "value_type": "float"
            },
            {
                "name": "bagging_freq",
                "type": "range",
                "bounds": [1, 7],
                "value_type": "int"
            },
            {
                "name": "min_child_samples",
                "type": "range",
                "bounds": [5, 100],
                "value_type": "int"
            },
            {
                "name": "lambda_l1",
                "type": "range",
                "bounds": [0, 10],
                "value_type": "float"
            },
            {
                "name": "lambda_l2",
                "type": "range",
                "bounds": [0, 10],
                "value_type": "float"
            },
            {
                "name": "max_depth",
                "type": "range",
                "bounds": [3, 12],
                "value_type": "int"
            },
            {
                "name": "min_gain_to_split",
                "type": "range",
                "bounds": [0.0, 15.0],
                "value_type": "float"
            }
        ]
    
    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare data for LightGBM training by encoding categorical features.
        
        LightGBM requires categorical features to be encoded as integers.
        """
        df_processed = df.copy()
        
        for cat_col in self.categorical_features:
            if df_processed[cat_col].dtype == 'object':
                if is_training:
                    # Create and store encoder for training
                    le = LabelEncoder()
                    df_processed[cat_col] = le.fit_transform(df_processed[cat_col])
                    self.categorical_encoders[cat_col] = le
                else:
                    # Use stored encoder for validation
                    if cat_col in self.categorical_encoders:
                        le = self.categorical_encoders[cat_col]
                        # Handle unseen categories by using -1
                        try:
                            df_processed[cat_col] = le.transform(df_processed[cat_col])
                        except ValueError:
                            # Handle unseen categories
                            seen_mask = df_processed[cat_col].isin(le.classes_)
                            df_processed.loc[seen_mask, cat_col] = le.transform(df_processed.loc[seen_mask, cat_col])
                            df_processed.loc[~seen_mask, cat_col] = -1
                    else:
                        # Fallback: create new encoder
                        le = LabelEncoder()
                        df_processed[cat_col] = le.fit_transform(df_processed[cat_col])
        
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
    ) -> lgb.Booster:
        """Train a LightGBM model with given parameters."""
        
        # Create LightGBM datasets (categorical features are already encoded)
        train_dataset = lgb.Dataset(
            X_train, 
            label=y_train, 
            weight=w_train,
            categorical_feature=self.categorical_features
        )
        val_dataset = lgb.Dataset(
            X_val, 
            label=y_val, 
            weight=w_val,
            categorical_feature=self.categorical_features,
            reference=train_dataset
        )
        
        # LightGBM parameters
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": 42,
            **params
        }
        
        # Train model
        model = lgb.train(
            lgb_params,
            train_dataset,
            valid_sets=[val_dataset],
            num_boost_round=max_iterations,
            callbacks=[lgb.log_evaluation(0)]  # Silent training
        )
        
        return model
    
    def predict(self, model: lgb.Booster, X: pd.DataFrame, iteration: Optional[int] = None) -> np.ndarray:
        """Make predictions using the trained LightGBM model."""
        if iteration is not None:
            # Predict with specific iteration
            return model.predict(X, num_iteration=iteration)
        else:
            # Predict with best iteration
            return model.predict(X)