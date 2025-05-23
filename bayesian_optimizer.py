import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy, ThresholdEarlyStoppingStrategy
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import wandb
from abc import ABC, abstractmethod

try:
    import catboost as cb
except ImportError:
    cb = None


class BayesianHyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization pipeline using Ax platform.
    Supports regression problems with k-fold cross-validation and early stopping.
    """
    
    def __init__(
        self,
        model_type: str,
        experiment_name: str,
        numerical_features: List[str],
        categorical_features: List[str],
        target_column: str,
        weight_column: str,
        fold_column: str,
        max_iterations: int,
        use_early_stopping: bool = True,
        early_stopping_strategy: str = "percentile",
        early_stopping_config: Optional[Dict] = None,
        iteration_early_stopping: bool = True,
        iteration_patience: int = 10,
        iteration_min_rounds: int = 20,
        output_dir: str = "experiments",
        use_wandb: bool = False,
        wandb_project: str = "bayesian-optimization",
        wandb_entity: Optional[str] = None
    ):
        # Validate model type
        if model_type not in ["lightgbm", "catboost"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'lightgbm' or 'catboost'")
        
        # Check if required library is available
        if model_type == "catboost" and cb is None:
            raise ImportError("catboost package is required for model_type='catboost'. Install with: pip install catboost")
        
        self.model_type = model_type
        self.experiment_name = experiment_name
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.weight_column = weight_column
        self.fold_column = fold_column
        self.max_iterations = max_iterations
        self.use_early_stopping = use_early_stopping
        self.early_stopping_strategy = early_stopping_strategy
        self.early_stopping_config = early_stopping_config or {}
        self.iteration_early_stopping = iteration_early_stopping
        self.iteration_patience = iteration_patience
        self.iteration_min_rounds = iteration_min_rounds
        
        # Wandb configuration
        self.use_wandb = use_wandb
        self.wandb_entity = wandb_entity
        self.current_trial_run = None     # Current trial run
        self.trial_runs = {}              # Store trial runs by index
        
        # Create unique experiment ID with timestamp
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # Set wandb project name to include timestamp for artifact correlation
        self.wandb_project = f"{wandb_project}_{self.experiment_id}" if wandb_project else None
        
        # Setup output directory
        self.output_dir = os.path.join(output_dir, self.experiment_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.ax_client = None
        self.df = None
        self.fold_values = None
        self.best_trial_index = None
        self.best_iteration = None
        self.categorical_encoders = {}
        
        # Trial runs will be created individually during optimization
        
    def _weighted_correlation(self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted correlation between predictions and target."""
        # Weighted means
        mean_true = np.average(y_true, weights=weights)
        mean_pred = np.average(y_pred, weights=weights)
        
        # Weighted covariances and variances
        cov = np.average((y_true - mean_true) * (y_pred - mean_pred), weights=weights)
        var_true = np.average((y_true - mean_true) ** 2, weights=weights)
        var_pred = np.average((y_pred - mean_pred) ** 2, weights=weights)
        
        # Correlation
        correlation = cov / np.sqrt(var_true * var_pred)
        return correlation
    
    def _init_wandb_trial(self, trial_index: int, parameters: Dict):
        """Initialize a separate wandb run for this trial."""
        if not self.use_wandb:
            return None
        
        try:
            # Create a new run for this trial
            trial_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=f"trial_{trial_index}_{self.experiment_id}",
                group=self.experiment_id,  # Group trials under experiment
                job_type="trial",
                config=parameters,
                tags=["bayesian-optimization", "trial", self.model_type],
                reinit=True
            )
            
            # Store the trial run
            self.trial_runs[trial_index] = trial_run
            self.current_trial_run = trial_run
            
            # Define metric types to avoid wandb auto-creating wrong panel types
            trial_run.define_metric("iteration")
            trial_run.define_metric("iteration_correlation", step_metric="iteration")
            trial_run.define_metric("best_correlation_so_far", step_metric="iteration")
            trial_run.define_metric("cv_std", step_metric="iteration")
            trial_run.define_metric("iterations_without_improvement", step_metric="iteration")
            
            # All metrics are now continuous or summary metrics
            
            print(f"Wandb trial {trial_index} initialized: {trial_run.url}")
            return trial_run
        except Exception as e:
            print(f"Warning: Failed to initialize trial wandb run: {e}")
            return None
    
    def _log_to_wandb(self, metrics: Dict, step: Optional[int] = None, trial_index: Optional[int] = None):
        """Log metrics to the current trial run."""
        if not self.use_wandb:
            return
        
        try:
            if trial_index is not None and trial_index in self.trial_runs:
                # Log to specific trial run
                self.trial_runs[trial_index].log(metrics, step=step)
            elif self.current_trial_run:
                # Log to current trial run
                self.current_trial_run.log(metrics, step=step)
            else:
                print("Warning: No active wandb run to log to")
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")
    
    def _finish_wandb_trial(self, trial_index: int):
        """Finish the specific trial run."""
        if not self.use_wandb or trial_index not in self.trial_runs:
            return
        
        try:
            # Finish the trial run
            self.trial_runs[trial_index].finish()
            
            # Remove from active trials
            del self.trial_runs[trial_index]
            
            # Clear current trial if it was this one
            if self.current_trial_run and trial_index in [int(name.split('_')[1]) for name in [self.current_trial_run.name] if '_' in name]:
                self.current_trial_run = None
                
            print(f"Wandb trial {trial_index} finished")
        except Exception as e:
            print(f"Warning: Failed to finish trial wandb run: {e}")
    
    def _create_search_space(self) -> List[Dict]:
        """Create hyperparameter search space based on model type."""
        if self.model_type == "lightgbm":
            return self._create_lightgbm_search_space()
        elif self.model_type == "catboost":
            return self._create_catboost_search_space()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _create_lightgbm_search_space(self) -> List[Dict]:
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
    
    def _create_catboost_search_space(self) -> List[Dict]:
        """Create CatBoost hyperparameter search space with comprehensive parameters."""
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
    
    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Encode categorical features consistently across train/val splits."""
        df_processed = df.copy()
        
        for cat_col in self.categorical_features:
            if df_processed[cat_col].dtype == 'object':
                if is_training:
                    # Create and store encoder for training
                    categories = df_processed[cat_col].astype('category').cat.categories
                    self.categorical_encoders[cat_col] = categories
                    df_processed[cat_col] = df_processed[cat_col].astype('category').cat.codes
                else:
                    # Use stored encoder for validation
                    if cat_col in self.categorical_encoders:
                        df_processed[cat_col] = df_processed[cat_col].astype('category').cat.set_categories(
                            self.categorical_encoders[cat_col]
                        ).cat.codes
                    else:
                        df_processed[cat_col] = df_processed[cat_col].astype('category').cat.codes
        
        return df_processed
    
    def _create_early_stopping_strategy(self):
        """Create the early stopping strategy based on configuration."""
        if not self.use_early_stopping:
            return None
        
        # Default configurations for each strategy type
        default_configs = {
            "percentile": {
                "percentile_threshold": 50.0,  # Stop bottom 50% of trials
                "min_progression": 3,  # Wait at least 3 iterations before stopping
                "min_curves": 3,  # Need at least 3 completed trials before stopping
                "normalize_progressions": True
            },
            "threshold": {
                "metric_threshold": 0.8,  # Stop if metric doesn't reach 0.8 by min_progression
                "min_progression": 5,  # Wait at least 5 iterations
                "min_curves": 3,  # Need at least 3 completed trials before stopping
                "normalize_progressions": True
            }
        }
        
        # Merge user config with defaults
        config = default_configs.get(self.early_stopping_strategy, {})
        config.update(self.early_stopping_config)
        
        if self.early_stopping_strategy == "percentile":
            return PercentileEarlyStoppingStrategy(
                metric_names=["weighted_correlation"],
                **config
            )
        elif self.early_stopping_strategy == "threshold":
            return ThresholdEarlyStoppingStrategy(
                metric_names=["weighted_correlation"],
                **config
            )
        else:
            raise ValueError(f"Unknown early stopping strategy: {self.early_stopping_strategy}")
    
    def setup_experiment(self, df: pd.DataFrame, search_space: Optional[List[Dict]] = None):
        """Setup the Ax experiment with the provided dataframe and search space."""
        self.df = df.copy()
        self.fold_values = sorted(df[self.fold_column].unique())
        
        # Handle categorical features based on model type
        if self.model_type == "lightgbm":
            # Pre-encode categorical features for LightGBM
            from sklearn.preprocessing import LabelEncoder
            for cat_col in self.categorical_features:
                if self.df[cat_col].dtype == 'object':
                    le = LabelEncoder()
                    self.df[cat_col] = le.fit_transform(self.df[cat_col])
                    self.categorical_encoders[cat_col] = le
        elif self.model_type == "catboost":
            # Keep categorical features as strings/objects for CatBoost native support
            # CatBoost handles categorical features natively, no encoding needed
            pass
        
        # Use default search space if none provided
        if search_space is None:
            search_space = self._create_search_space()
        
        # Create early stopping strategy
        early_stopping_strategy = self._create_early_stopping_strategy()
        
        # Initialize Ax client with early stopping strategy
        self.ax_client = AxClient(early_stopping_strategy=early_stopping_strategy)
        
        # Create experiment
        self.ax_client.create_experiment(
            name=self.experiment_id,
            parameters=search_space,
            objectives={"weighted_correlation": ObjectiveProperties(minimize=False)},
            tracking_metric_names=["weighted_correlation"] if self.use_early_stopping else [],
            support_intermediate_data=self.use_early_stopping,  # Required for early stopping
        )
        
        # Dataset statistics will be logged with each trial run
    
    def _train_fold_models(self, params: Dict, iteration: Optional[int] = None) -> Tuple[List, List[float]]:
        """Train models for all folds and return models and validation metrics."""
        if self.model_type == "lightgbm":
            return self._train_lightgbm_fold_models(params, iteration)
        elif self.model_type == "catboost":
            return self._train_catboost_fold_models(params, iteration)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _train_lightgbm_fold_models(self, params: Dict, iteration: Optional[int] = None) -> Tuple[List, List[float]]:
        """Train LightGBM models for all folds and return models and validation metrics."""
        models = []
        fold_metrics = []
        
        for fold in self.fold_values:
            # Split data
            train_data = self.df[self.df[self.fold_column] != fold]
            val_data = self.df[self.df[self.fold_column] == fold]
            
            # Prepare features
            all_features = self.numerical_features + self.categorical_features
            X_train = train_data[all_features]
            y_train = train_data[self.target_column]
            w_train = train_data[self.weight_column]
            
            X_val = val_data[all_features]
            y_val = val_data[self.target_column]
            w_val = val_data[self.weight_column]
            
            # Create LightGBM datasets (categorical features are already encoded)
            train_dataset = lgb.Dataset(
                X_train, label=y_train, weight=w_train,
                categorical_feature=self.categorical_features
            )
            val_dataset = lgb.Dataset(
                X_val, label=y_val, weight=w_val,
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
                num_boost_round=self.max_iterations,
                callbacks=[lgb.log_evaluation(0)]  # Silent training
            )
            
            # Predict on validation set
            if iteration is not None:
                # Predict with specific iteration
                y_pred = model.predict(X_val, num_iteration=iteration)
            else:
                # Predict with best iteration
                y_pred = model.predict(X_val)
            
            # Calculate weighted correlation
            correlation = self._weighted_correlation(y_val.values, y_pred, w_val.values)
            
            models.append(model)
            fold_metrics.append(correlation)
        
        return models, fold_metrics
    
    def _train_catboost_fold_models(self, params: Dict, iteration: Optional[int] = None) -> Tuple[List, List[float]]:
        """Train CatBoost models for all folds and return models and validation metrics."""
        models = []
        fold_metrics = []
        
        for fold in self.fold_values:
            # Split data
            train_data = self.df[self.df[self.fold_column] != fold]
            val_data = self.df[self.df[self.fold_column] == fold]
            
            # Prepare features
            all_features = self.numerical_features + self.categorical_features
            X_train = train_data[all_features]
            y_train = train_data[self.target_column]
            w_train = train_data[self.weight_column]
            
            X_val = val_data[all_features]
            y_val = val_data[self.target_column]
            w_val = val_data[self.weight_column]
            
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
                "iterations": self.max_iterations,
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
            
            # Predict on validation set
            if iteration is not None:
                # Predict with specific iteration (CatBoost uses ntree_end)
                y_pred = model.predict(X_val, ntree_end=iteration)
            else:
                # Predict with all iterations
                y_pred = model.predict(X_val)
            
            # Calculate weighted correlation
            correlation = self._weighted_correlation(y_val.values, y_pred, w_val.values)
            
            models.append(model)
            fold_metrics.append(correlation)
        
        return models, fold_metrics
    
    def _evaluate_trial_with_progression(self, trial_index: int, parameters: Dict) -> None:
        """Evaluate a trial with both iteration-level and trial-level early stopping."""
        # Initialize wandb trial run
        trial_run = self._init_wandb_trial(trial_index, parameters)
        
        # Train all fold models first (full training)
        models, _ = self._train_fold_models(parameters)
        
        # Track best iteration within this trial
        best_correlation = -float('inf')
        best_iteration = 1
        iterations_without_improvement = 0
        
        # Evaluate each iteration progressively
        for iteration in range(1, self.max_iterations + 1):
            fold_metrics = []
            
            for fold_idx, fold in enumerate(self.fold_values):
                # Get validation data for this fold
                val_data = self.df[self.df[self.fold_column] == fold]
                all_features = self.numerical_features + self.categorical_features
                X_val = val_data[all_features]
                y_val = val_data[self.target_column]
                w_val = val_data[self.weight_column]
                
                # Predict with specific iteration using model-specific interface
                if self.model_type == "lightgbm":
                    y_pred = models[fold_idx].predict(X_val, num_iteration=iteration)
                elif self.model_type == "catboost":
                    y_pred = models[fold_idx].predict(X_val, ntree_end=iteration)
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}")
                correlation = self._weighted_correlation(y_val.values, y_pred, w_val.values)
                fold_metrics.append(correlation)
            
            mean_correlation = np.mean(fold_metrics)
            std_correlation = np.std(fold_metrics)
            
            # Track best iteration within this trial (iteration-level early stopping)
            if mean_correlation > best_correlation:
                best_correlation = mean_correlation
                best_iteration = iteration
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            # Log iteration metrics to wandb
            if self.use_wandb:
                iteration_metrics = {
                    "iteration_correlation": mean_correlation,
                    "best_correlation_so_far": best_correlation,
                    "iterations_without_improvement": iterations_without_improvement,
                    "cv_std": std_correlation,
                    "iteration": iteration
                }
                # Log per-fold correlations
                for fold_idx, fold_corr in enumerate(fold_metrics):
                    iteration_metrics[f"fold_{self.fold_values[fold_idx]}_correlation"] = fold_corr
                
                self._log_to_wandb(iteration_metrics, step=iteration, trial_index=trial_index)
            
            # Attach progressive data to Ax with progression metadata
            raw_data = [
                ({"iteration": iteration}, {"weighted_correlation": (mean_correlation, 0.0)})
            ]
            self.ax_client.update_running_trial_with_intermediate_data(
                trial_index=trial_index,
                raw_data=raw_data
            )
            
            # Iteration-level early stopping: stop if no improvement for patience iterations
            if (self.iteration_early_stopping and 
                iterations_without_improvement >= self.iteration_patience and 
                iteration > self.iteration_min_rounds):
                print(f"  Trial {trial_index} stopped at iteration {iteration} (no improvement for {self.iteration_patience} iterations)")
                print(f"  Best iteration for trial {trial_index}: {best_iteration} (correlation: {best_correlation:.4f})")
                
                # Log iteration-level early stopping to wandb
                if self.use_wandb:
                    early_stop_metrics = {
                        "final_iteration": iteration,
                        "best_iteration": best_iteration
                    }
                    self._log_to_wandb(early_stop_metrics, trial_index=trial_index)
                
                # IMPORTANT: Report the BEST correlation to Ax for Bayesian optimization
                # This ensures Ax compares trials based on their optimal performance
                best_raw_data = [
                    ({"iteration": best_iteration}, {"weighted_correlation": (best_correlation, 0.0)})
                ]
                self.ax_client.update_running_trial_with_intermediate_data(
                    trial_index=trial_index,
                    raw_data=best_raw_data
                )
                break
            
            # Trial-level early stopping: check if trial should be stopped early by Ax
            if self.use_early_stopping and iteration >= 3:  # Wait minimum iterations
                should_stop = self.ax_client.should_stop_trials_early(trial_indices=[trial_index])
                if should_stop and should_stop.get(trial_index, False):
                    print(f"  Trial {trial_index} stopped early by Ax at iteration {iteration}")
                    
                    # Log Ax-level early stopping to wandb
                    if self.use_wandb:
                        ax_stop_metrics = {
                            "final_iteration": iteration,
                            "best_iteration": best_iteration
                        }
                        self._log_to_wandb(ax_stop_metrics, trial_index=trial_index)
                    
                    self.ax_client.stop_trial_early(trial_index=trial_index)
                    break
        
        # Store the best iteration for this trial
        if not hasattr(self, 'trial_best_iterations'):
            self.trial_best_iterations = {}
        self.trial_best_iterations[trial_index] = best_iteration
        
        # If trial completed without early stopping, report the best correlation to Ax
        if iteration == self.max_iterations:
            print(f"  Trial {trial_index} completed all {self.max_iterations} iterations")
            print(f"  Best iteration: {best_iteration} (correlation: {best_correlation:.4f})")
            
            # Report the BEST correlation to Ax for Bayesian optimization
            best_raw_data = [
                ({"iteration": best_iteration}, {"weighted_correlation": (best_correlation, 0.0)})
            ]
            self.ax_client.update_running_trial_with_intermediate_data(
                trial_index=trial_index,
                raw_data=best_raw_data
            )
        
        # Log final trial summary to wandb
        if self.use_wandb:
            trial_summary = {
                "final_best_correlation": best_correlation,
                "best_iteration": best_iteration,
                "total_iterations": iteration
            }
            self._log_to_wandb(trial_summary, trial_index=trial_index)
            
            # Finish the trial run
            self._finish_wandb_trial(trial_index)
        
        # Mark trial as completed (without additional raw_data since we used intermediate updates)
        try:
            # For trials with intermediate data, complete without additional raw_data
            trial = self.ax_client.experiment.trials[trial_index] 
            if trial.status.is_running:
                trial.mark_completed()
            print(f"  Trial {trial_index} completed")
        except Exception as e:
            print(f"  Warning: Could not mark trial {trial_index} as completed: {e}")
            print(f"  Trial {trial_index} finished")
    
    def _evaluate_trial_without_progression(self, parameters: Dict) -> Dict[str, float]:
        """Evaluate trial without early stopping - use final iteration."""
        models, fold_metrics = self._train_fold_models(parameters)
        mean_correlation = np.mean(fold_metrics)
        return {"weighted_correlation": mean_correlation}
    
    def optimize(self, total_trials: int = 50) -> Dict:
        """Run the optimization loop."""
        # Optimization tracking is done in individual trial runs
        
        for i in range(total_trials):
            # Get next trial
            parameters, trial_index = self.ax_client.get_next_trial()
            
            print(f"Trial {i+1}/{total_trials} (index={trial_index}): Starting evaluation...")
            
            if self.use_early_stopping:
                # Use progression-based evaluation with native Ax early stopping
                self._evaluate_trial_with_progression(trial_index, parameters)
                
                # The best correlation was already reported to Ax during trial evaluation
                print(f"  Trial {trial_index} evaluation completed")
                
                # Trial completion is tracked in individual trial runs
            else:
                # Use simple evaluation without progression
                results = self._evaluate_trial_without_progression(parameters)
                
                # Complete trial
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=results)
                
                correlation = results['weighted_correlation']
                if isinstance(correlation, tuple):
                    correlation = correlation[0]
                print(f"  Final correlation: {correlation:.4f}")
        
        # Get best trial
        result = self.ax_client.get_best_trial()
        if result is not None:
            self.best_trial_index, best_parameters, predictions = result
            best_values = predictions[0]['weighted_correlation'] if predictions else 0.0
        else:
            self.best_trial_index = None
            best_parameters = {}
            best_values = 0.0
        
        # Handle tuple format from Ax
        if isinstance(best_values, tuple):
            best_values = best_values[0]
        
        print(f"Best trial: {self.best_trial_index}")
        print(f"Best parameters: {best_parameters}")
        print(f"Best weighted correlation: {best_values:.4f}")
        
        # Experiment summary is available in individual trial runs
        
        return {
            "best_parameters": best_parameters,
            "best_correlation": best_values,
            "best_trial_index": self.best_trial_index
        }
    
    def save_best_model_artifacts(self):
        """Save the best models and predictions."""
        if self.best_trial_index is None:
            raise ValueError("No optimization has been run yet")
        
        # Get best trial parameters
        best_trial = self.ax_client.experiment.trials[self.best_trial_index]
        best_parameters = best_trial.arm.parameters
        
        # Train final models with best parameters
        models, _ = self._train_fold_models(best_parameters)
        
        # Use the stored best iteration from the trial evaluation
        if self.use_early_stopping and hasattr(self, 'trial_best_iterations'):
            self.best_iteration = self.trial_best_iterations.get(self.best_trial_index, self.max_iterations)
        else:
            self.best_iteration = self.max_iterations
        
        # Save models
        models_path = os.path.join(self.output_dir, "models.pkl")
        with open(models_path, 'wb') as f:
            pickle.dump(models, f)
        
        # Save iteration info
        iteration_info = {
            "model_type": self.model_type,
            "best_iteration": self.best_iteration,
            "best_parameters": best_parameters,
            "experiment_id": self.experiment_id
        }
        iteration_path = os.path.join(self.output_dir, "iteration_info.pkl")
        with open(iteration_path, 'wb') as f:
            pickle.dump(iteration_info, f)
        
        # Generate and save predictions
        predictions_df = self.df.copy()
        predictions_df['prediction'] = 0.0
        
        for fold_idx, fold in enumerate(self.fold_values):
            fold_mask = self.df[self.fold_column] == fold
            val_data = self.df[fold_mask]
            all_features = self.numerical_features + self.categorical_features
            X_val = val_data[all_features]
            
            # Predict with best iteration using model-specific interface
            if self.model_type == "lightgbm":
                y_pred = models[fold_idx].predict(X_val, num_iteration=self.best_iteration)
            elif self.model_type == "catboost":
                y_pred = models[fold_idx].predict(X_val, ntree_end=self.best_iteration)
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
            predictions_df.loc[fold_mask, 'prediction'] = y_pred
        
        # Save predictions
        predictions_path = os.path.join(self.output_dir, "predictions.pkl")
        predictions_df.to_pickle(predictions_path)
        
        print(f"Artifacts saved to: {self.output_dir}")
        print(f"- Models: models.pkl")
        print(f"- Iteration info: iteration_info.pkl") 
        print(f"- Predictions: predictions.pkl")
        
        # Save artifacts to wandb
        if self.use_wandb:
            try:
                # Get best correlation from trial results
                best_result = self.ax_client.get_best_trial()
                best_correlation = 0.0
                if best_result and best_result[2]:
                    best_correlation = best_result[2][0]['weighted_correlation']
                    if isinstance(best_correlation, tuple):
                        best_correlation = best_correlation[0]
                
                # Create wandb artifact for the best model
                model_artifact = wandb.Artifact(
                    name=f"best_model_{self.experiment_id}",
                    type="model",
                    description=f"Best {self.model_type.upper()} model from trial {self.best_trial_index} at iteration {self.best_iteration}",
                    metadata={
                        "model_type": self.model_type,
                        "best_trial_index": self.best_trial_index,
                        "best_iteration": self.best_iteration,
                        "best_correlation": float(best_correlation),
                        "experiment_id": self.experiment_id
                    }
                )
                
                # Add model files to artifact
                model_artifact.add_file(models_path, name="models.pkl")
                model_artifact.add_file(iteration_path, name="iteration_info.pkl")
                model_artifact.add_file(predictions_path, name="predictions.pkl")
                
                # Log artifact to experiment run
                self.wandb_experiment_run.log_artifact(model_artifact)
                print("Model artifacts logged to wandb")
                
            except Exception as e:
                print(f"Warning: Could not save artifacts to wandb: {e}")
        
        return {
            "output_dir": self.output_dir,
            "best_iteration": self.best_iteration,
            "models_path": models_path,
            "predictions_path": predictions_path
        }
    
    def finish_wandb(self):
        """Finish all wandb runs and cleanup."""
        if self.use_wandb:
            try:
                # Finish any remaining trial runs
                for trial_index in list(self.trial_runs.keys()):
                    self._finish_wandb_trial(trial_index)
                print("All wandb trial runs finished")
            except Exception as e:
                print(f"Warning: Error finishing wandb runs: {e}")