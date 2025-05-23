"""
Random Forest learner implementation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from .base_learner import BaseLearner


class RandomForestLearner(BaseLearner):
    """
    Random Forest learner implementation for Bayesian optimization.
    
    Handles Random Forest-specific training, prediction, and hyperparameter search space.
    """
    
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        super().__init__(numerical_features, categorical_features)
        self.categorical_encoders = {}
    
    @property
    def name(self) -> str:
        return "sklearn_random_forest"
    
    def supports_iteration_prediction(self) -> bool:
        """Random Forest doesn't support iteration-specific predictions."""
        return False
    
    def get_default_max_iterations(self) -> int:
        """Random Forest uses n_estimators instead of iterations."""
        return 200  # Default number of trees
    
    def get_search_space(self) -> List[Dict]:
        """Create Random Forest hyperparameter search space."""
        return [
            {
                "name": "n_estimators",
                "type": "range",
                "bounds": [50, 500],
                "value_type": "int"
            },
            {
                "name": "max_depth",
                "type": "range",
                "bounds": [3, 20],
                "value_type": "int"
            },
            {
                "name": "min_samples_split",
                "type": "range",
                "bounds": [2, 20],
                "value_type": "int"
            },
            {
                "name": "min_samples_leaf",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int"
            },
            {
                "name": "max_features",
                "type": "choice",
                "values": ["sqrt", "log2"],
                "value_type": "str"
            },
            {
                "name": "bootstrap",
                "type": "choice",
                "values": [True, False],
                "value_type": "bool"
            },
            {
                "name": "min_impurity_decrease",
                "type": "range",
                "bounds": [0.0, 0.1],
                "value_type": "float"
            }
        ]
    
    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare data for Random Forest training by encoding categorical features.
        
        sklearn's Random Forest requires all features to be numeric.
        """
        df_processed = df.copy()
        
        for cat_col in self.categorical_features:
            if df_processed[cat_col].dtype == 'object' or df_processed[cat_col].dtype.name == 'category':
                if is_training:
                    # Create and store encoder for training
                    le = LabelEncoder()
                    df_processed[cat_col] = le.fit_transform(df_processed[cat_col].astype(str))
                    self.categorical_encoders[cat_col] = le
                else:
                    # Use stored encoder for validation
                    if cat_col in self.categorical_encoders:
                        le = self.categorical_encoders[cat_col]
                        # Handle unseen categories by using -1
                        try:
                            df_processed[cat_col] = le.transform(df_processed[cat_col].astype(str))
                        except ValueError:
                            # Handle unseen categories
                            seen_mask = df_processed[cat_col].astype(str).isin(le.classes_)
                            df_processed.loc[seen_mask, cat_col] = le.transform(df_processed.loc[seen_mask, cat_col].astype(str))
                            df_processed.loc[~seen_mask, cat_col] = -1
                    else:
                        # Fallback: create new encoder
                        le = LabelEncoder()
                        df_processed[cat_col] = le.fit_transform(df_processed[cat_col].astype(str))
            
            # Ensure categorical columns are numeric
            df_processed[cat_col] = pd.to_numeric(df_processed[cat_col], errors='coerce').fillna(-1).astype(int)
        
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
    ) -> RandomForestRegressor:
        """Train a Random Forest model with given parameters."""
        
        # For Random Forest, max_iterations maps to n_estimators if not specified in params
        rf_params = {
            "random_state": 42,
            "n_jobs": -1,  # Use all available cores
            **params
        }
        
        # If n_estimators not in params, use max_iterations
        if "n_estimators" not in rf_params:
            rf_params["n_estimators"] = max_iterations
        
        # Create and train model
        model = RandomForestRegressor(**rf_params)
        
        # Convert pandas DataFrames/Series to numpy arrays for sklearn
        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        w_train_array = w_train.values if hasattr(w_train, 'values') else w_train
        
        model.fit(X_train_array, y_train_array, sample_weight=w_train_array)
        
        return model
    
    def predict(self, model: RandomForestRegressor, X: pd.DataFrame, iteration: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.
        
        Note: iteration parameter is ignored for Random Forest as it doesn't support
        iteration-specific predictions.
        """
        if iteration is not None:
            # Log warning that iteration is ignored
            print(f"Warning: Random Forest doesn't support iteration-specific predictions. Ignoring iteration={iteration}")
        
        # Convert pandas DataFrame to numpy array for sklearn
        X_array = X.values if hasattr(X, 'values') else X
        return model.predict(X_array)