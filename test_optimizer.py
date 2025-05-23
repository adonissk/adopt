import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from sklearn.datasets import make_regression
from bayesian_optimizer import BayesianHyperparameterOptimizer


class TestBayesianHyperparameterOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample dataset
        X, y = make_regression(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        
        # Create dataframe
        feature_names = [f'num_feature_{i}' for i in range(8)] + ['cat_feature_1', 'cat_feature_2']
        self.df = pd.DataFrame(X, columns=feature_names)
        
        # Convert some features to categorical
        self.df['cat_feature_1'] = (self.df['cat_feature_1'] > 0).astype(str)
        self.df['cat_feature_2'] = pd.cut(self.df['cat_feature_2'], bins=3, labels=['A', 'B', 'C'])
        
        # Add target, weight, and fold
        self.df['target'] = y
        self.df['weight'] = np.random.uniform(0.5, 2.0, 1000)
        self.df['fold'] = np.random.randint(1, 4, 1000)  # 3 folds
        
        # Define features
        self.numerical_features = [f'num_feature_{i}' for i in range(8)]
        self.categorical_features = ['cat_feature_1', 'cat_feature_2']
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test_experiment",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight",
            fold_column="fold",
            max_iterations=50,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        self.assertEqual(optimizer.experiment_name, "test_experiment")
        self.assertEqual(optimizer.numerical_features, self.numerical_features)
        self.assertEqual(optimizer.categorical_features, self.categorical_features)
        self.assertEqual(optimizer.target_column, "target")
        self.assertEqual(optimizer.weight_column, "weight")
        self.assertEqual(optimizer.fold_column, "fold")
        self.assertEqual(optimizer.max_iterations, 50)
        self.assertFalse(optimizer.use_early_stopping)
        self.assertTrue(optimizer.experiment_id.startswith("test_experiment_"))
    
    def test_weighted_correlation(self):
        """Test weighted correlation calculation."""
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight", 
            fold_column="fold",
            max_iterations=10,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        # Test perfect correlation
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        weights = np.array([1, 1, 1, 1, 1])
        
        correlation = optimizer._weighted_correlation(y_true, y_pred, weights)
        self.assertAlmostEqual(correlation, 1.0, places=5)
        
        # Test no correlation
        y_pred_random = np.array([5, 1, 4, 2, 3])
        correlation = optimizer._weighted_correlation(y_true, y_pred_random, weights)
        self.assertLess(abs(correlation), 0.5)  # Should be low correlation
    
    def test_search_space_creation(self):
        """Test LightGBM search space creation."""
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight",
            fold_column="fold", 
            max_iterations=10,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        search_space = optimizer._create_search_space()
        
        # Check that we have expected parameters
        param_names = [param['name'] for param in search_space]
        expected_params = ['num_leaves', 'learning_rate', 'feature_fraction', 
                          'bagging_fraction', 'bagging_freq', 'min_child_samples',
                          'lambda_l1', 'lambda_l2', 'max_depth', 'min_gain_to_split']
        
        for param in expected_params:
            self.assertIn(param, param_names)
        
        # Check parameter types and bounds
        for param in search_space:
            self.assertIn('name', param)
            self.assertIn('type', param)
            self.assertIn('bounds', param)
            self.assertIn('value_type', param)
            self.assertEqual(param['type'], 'range')
            self.assertEqual(len(param['bounds']), 2)
    
    def test_experiment_setup(self):
        """Test experiment setup."""
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight",
            fold_column="fold",
            max_iterations=10,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        optimizer.setup_experiment(self.df)
        
        self.assertIsNotNone(optimizer.ax_client)
        self.assertIsNotNone(optimizer.df)
        self.assertEqual(len(optimizer.fold_values), 3)  # We have 3 folds
        self.assertEqual(optimizer.fold_values, [1, 2, 3])
    
    def test_fold_model_training(self):
        """Test training models for each fold."""
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight",
            fold_column="fold",
            max_iterations=10,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        optimizer.setup_experiment(self.df)
        
        # Test parameters
        test_params = {
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 0,
            "lambda_l2": 0,
            "max_depth": -1,
            "min_gain_to_split": 0
        }
        
        models, fold_metrics = optimizer._train_fold_models(test_params)
        
        self.assertEqual(len(models), 3)  # One model per fold
        self.assertEqual(len(fold_metrics), 3)  # One metric per fold
        
        # Check that all metrics are reasonable correlations
        for metric in fold_metrics:
            self.assertTrue(-1 <= metric <= 1)
    
    def test_optimization_without_early_stopping(self):
        """Test optimization without early stopping."""
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test_no_early_stopping",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight",
            fold_column="fold",
            max_iterations=10,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        optimizer.setup_experiment(self.df)
        
        # Run very short optimization
        results = optimizer.optimize(total_trials=2)
        
        self.assertIn('best_parameters', results)
        self.assertIn('best_correlation', results)
        self.assertIn('best_trial_index', results)
        
        # Check that correlation is reasonable
        self.assertTrue(-1 <= results['best_correlation'] <= 1)
    
    def test_artifact_saving(self):
        """Test saving of model artifacts."""
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test_artifacts",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight",
            fold_column="fold",
            max_iterations=10,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        optimizer.setup_experiment(self.df)
        results = optimizer.optimize(total_trials=2)
        artifacts = optimizer.save_best_model_artifacts()
        
        # Check that files were created
        self.assertTrue(os.path.exists(artifacts['models_path']))
        self.assertTrue(os.path.exists(artifacts['predictions_path']))
        
        # Check that predictions file contains expected columns
        predictions_df = pd.read_pickle(artifacts['predictions_path'])
        self.assertIn('prediction', predictions_df.columns)
        self.assertEqual(len(predictions_df), len(self.df))
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test saving artifacts before optimization
        optimizer = BayesianHyperparameterOptimizer(
            experiment_name="test_error",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            target_column="target",
            weight_column="weight",
            fold_column="fold",
            max_iterations=10,
            use_early_stopping=False,
            output_dir=self.test_dir
        )
        
        with self.assertRaises(ValueError):
            optimizer.save_best_model_artifacts()


if __name__ == '__main__':
    unittest.main()