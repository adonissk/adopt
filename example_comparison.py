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


def run_optimization(model_type: str, df: pd.DataFrame, total_trials: int = 10):
    """Run optimization for a specific model type."""
    print(f"\n{'='*50}")
    print(f"Running {model_type.upper()} Optimization")
    print(f"{'='*50}")
    
    # Define features
    numerical_features = [col for col in df.columns if col.startswith('num_feature_')]
    categorical_features = ['cat_feature_1', 'cat_feature_2', 'cat_feature_3']
    
    # Trial-level early stopping config
    trial_early_stopping_config = {
        "percentile_threshold": 25.0,
        "min_progression": 5,  # Reduced for comparison
        "min_curves": 3,
    }
    
    optimizer = BayesianHyperparameterOptimizer(
        model_type=model_type,
        experiment_name=f"{model_type}_comparison_example",
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column="target",
        weight_column="weight",
        fold_column="fold",
        max_iterations=50,  # Reduced for comparison
        
        # Trial-level early stopping
        use_early_stopping=True,
        early_stopping_strategy="percentile",
        early_stopping_config=trial_early_stopping_config,
        
        # Iteration-level early stopping
        iteration_early_stopping=True,
        iteration_patience=10,  # Reduced for comparison
        iteration_min_rounds=15,  # Reduced for comparison
        
        output_dir="experiments",
        
        # Wandb logging
        use_wandb=True,
        wandb_project=f"{model_type}-comparison",
        wandb_entity=None
    )
    
    print(f"Experiment ID: {optimizer.experiment_id}")
    
    # Setup and run
    optimizer.setup_experiment(df)
    results = optimizer.optimize(total_trials=total_trials)
    
    # Save artifacts
    artifacts = optimizer.save_best_model_artifacts()
    
    # Finish wandb
    optimizer.finish_wandb()
    
    return {
        "model_type": model_type,
        "best_correlation": results['best_correlation'],
        "best_parameters": results['best_parameters'],
        "best_trial_index": results['best_trial_index'],
        "experiment_id": optimizer.experiment_id,
        "output_dir": artifacts['output_dir'],
        "best_iteration": artifacts['best_iteration']
    }


def main():
    """Compare LightGBM and CatBoost performance on the same dataset."""
    print("Creating sample dataset for comparison...")
    df = create_sample_dataset(n_samples=3000, n_features=12, n_folds=5)  # Smaller for comparison
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fold distribution: {df['fold'].value_counts().sort_index()}")
    
    # Run optimizations for both models
    results = []
    
    for model_type in ["lightgbm", "catboost"]:
        try:
            result = run_optimization(model_type, df, total_trials=8)  # Reduced trials for comparison
            results.append(result)
        except Exception as e:
            print(f"Error running {model_type} optimization: {e}")
            continue
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        for result in results:
            print(f"\n{result['model_type'].upper()} Results:")
            print(f"  Best correlation: {result['best_correlation']:.4f}")
            print(f"  Best iteration: {result['best_iteration']}")
            print(f"  Best trial: {result['best_trial_index']}")
            print(f"  Experiment ID: {result['experiment_id']}")
            print(f"  Output directory: {result['output_dir']}")
        
        # Determine winner
        best_result = max(results, key=lambda x: x['best_correlation'])
        print(f"\n🏆 WINNER: {best_result['model_type'].upper()}")
        print(f"   Best correlation: {best_result['best_correlation']:.4f}")
        
        # Performance difference
        correlations = [r['best_correlation'] for r in results]
        diff = max(correlations) - min(correlations)
        print(f"   Performance difference: {diff:.4f}")
        
    else:
        print("Could not complete comparison - insufficient results")
        for result in results:
            print(f"{result['model_type'].upper()}: {result['best_correlation']:.4f}")


if __name__ == "__main__":
    main()