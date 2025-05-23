import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from bayesian_optimizer import BayesianHyperparameterOptimizer
import os 
import wandb 

os.environ["WANDB_DIR"] = "/tmp"
os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"
os.environ["WANDB_API_KEY"] = "local-3c0791a415f876256853d1e5f2ec1ad823981035"


def create_sample_dataset(n_samples=10000, n_features=20, n_folds=5):
    """Create a sample regression dataset with fold column."""
    # Generate regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        noise=0.1,
        random_state=42
    )
    
    # Create dataframe
    feature_names = [f'num_feature_{i}' for i in range(n_features-3)] + ['cat_feature_1', 'cat_feature_2', 'cat_feature_3']
    df = pd.DataFrame(X, columns=feature_names)
    
    # Convert some features to categorical
    df['cat_feature_1'] = (df['cat_feature_1'] > 0).astype(str)
    df['cat_feature_2'] = pd.cut(df['cat_feature_2'], bins=5, labels=['A', 'B', 'C', 'D', 'E'])
    df['cat_feature_3'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
    
    # Add target and weight
    df['target'] = y
    df['weight'] = np.random.uniform(0.5, 2.0, n_samples)  # Random weights
    
    # Add fold column
    df['fold'] = np.random.randint(1, n_folds + 1, n_samples)
    
    return df


def main():
    """Example usage of the Bayesian Hyperparameter Optimizer with Random Forest."""
    print("Creating sample dataset...")
    df = create_sample_dataset(n_samples=5000, n_features=15, n_folds=5)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fold distribution: {df['fold'].value_counts().sort_index()}")
    
    # Define features
    numerical_features = [col for col in df.columns if col.startswith('num_feature_')]
    categorical_features = ['cat_feature_1', 'cat_feature_2', 'cat_feature_3']
    
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Initialize optimizer with Random Forest
    # Note: Random Forest doesn't support iteration-level early stopping, 
    # so we use fewer trials and disable iteration early stopping
    
    optimizer = BayesianHyperparameterOptimizer(
        model_type="sklearn_random_forest",  # New model type!
        experiment_name="random_forest_example",
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column="target",
        weight_column="weight",
        fold_column="fold",
        max_iterations=200,  # This maps to n_estimators for Random Forest
        
        # Early stopping configuration
        use_early_stopping=True,  # Enable for full wandb logging
        iteration_early_stopping=False,  # Random Forest doesn't support iteration prediction
        
        output_dir="experiments",
        
        # Wandb logging
        use_wandb=True,
        wandb_project="random-forest-bayesian-optimization",
        wandb_entity=None
    )
    
    print(f"Experiment ID: {optimizer.experiment_id}")
    print(f"Using learner: {optimizer.learner.name}")
    
    # Setup experiment with Random Forest search space
    print("Setting up experiment...")
    optimizer.setup_experiment(df)
    
    # Run optimization
    print("Starting optimization...")
    results = optimizer.optimize(total_trials=5)  # Fewer trials for demonstration
    
    print("\nOptimization Results:")
    print(f"Best correlation: {results['best_correlation']:.4f}")
    print(f"Best parameters: {results['best_parameters']}")
    
    # Save artifacts
    print("\nSaving model artifacts...")
    artifacts = optimizer.save_best_model_artifacts()
    
    print(f"All artifacts saved to: {artifacts['output_dir']}")
    print(f"Best iteration used: {artifacts['best_iteration']}")
    
    # Finish wandb runs
    optimizer.finish_wandb()
    
    # Show the capabilities of the new system
    print(f"\nLearner Capabilities:")
    print(f"- Supports iteration prediction: {optimizer.learner.supports_iteration_prediction()}")
    print(f"- Default max iterations: {optimizer.learner.get_default_max_iterations()}")
    
    print(f"\nWandB Logging:")
    print(f"- Random Forest logs the same core metrics as LightGBM/CatBoost")
    print(f"- Since RF doesn't support iterations, only 1 data point per trial")
    print(f"- All trials are tracked with correlation metrics and fold-level details")


if __name__ == "__main__":
    main()