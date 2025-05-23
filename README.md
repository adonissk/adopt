# Bayesian Hyperparameter Optimization Pipeline

A comprehensive Bayesian hyperparameter optimization pipeline built with Ax platform for machine learning models, specifically designed for regression problems with k-fold cross-validation.

## Features

- **Bayesian Optimization**: Uses Facebook's Ax platform with BoTorch for efficient hyperparameter search
- **K-Fold Cross-Validation**: Robust evaluation using custom fold columns
- **Native Ax Early Stopping**: Uses Ax's built-in early stopping strategies (Percentile and Threshold)
- **Progressive Evaluation**: Iteration-by-iteration model evaluation with early termination
- **Weighted Metrics**: Supports weighted correlation as the optimization objective
- **Model Persistence**: Saves best models, predictions, and optimization artifacts
- **Extensible Design**: Modular architecture for easy extension to other ML frameworks

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from bayesian_optimizer import BayesianHyperparameterOptimizer
import pandas as pd

# Your data should have: features, target, weight, and fold columns
# df = pd.read_csv('your_data.csv')

# Define feature columns
numerical_features = ['feature1', 'feature2', 'feature3']
categorical_features = ['cat_feature1', 'cat_feature2']

# Configure both levels of early stopping
trial_early_stopping_config = {
    "percentile_threshold": 25.0,  # Stop bottom 25% of trials
    "min_progression": 10,  # Wait at least 10 iterations before stopping trials
    "min_curves": 3,  # Need at least 3 completed trials before stopping
}

# Initialize optimizer with dual early stopping
optimizer = BayesianHyperparameterOptimizer(
    experiment_name="my_experiment",
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target_column="target",
    weight_column="weight",
    fold_column="fold",
    max_iterations=200,  # Maximum iterations per trial
    
    # Trial-level early stopping (between trials)
    use_early_stopping=True,
    early_stopping_strategy="percentile",  # or "threshold"
    early_stopping_config=trial_early_stopping_config,
    
    # Iteration-level early stopping (within trials)
    iteration_early_stopping=True,
    iteration_patience=15,  # Stop if no improvement for 15 iterations
    iteration_min_rounds=30,  # Minimum 30 iterations before early stopping
)

# Setup and run optimization
optimizer.setup_experiment(df)
results = optimizer.optimize(total_trials=50)

# Save best models and predictions
artifacts = optimizer.save_best_model_artifacts()
```

## Data Requirements

Your dataset should contain:

1. **Numerical Features**: Float/integer columns for numerical features
2. **Categorical Features**: String/categorical columns (will be automatically encoded)
3. **Target Column**: The regression target variable
4. **Weight Column**: Sample weights for weighted correlation calculation
5. **Fold Column**: Integer values (1, 2, 3, ..., k) defining cross-validation folds

## How It Works

### Cross-Validation Strategy
- For k folds, trains k models where each model uses (k-1) folds for training and 1 fold for validation
- Example: For fold 1, training data = `df[df.fold != 1]`, validation data = `df[df.fold == 1]`

### Dual-Level Early Stopping
The pipeline implements **both** iteration-level and trial-level early stopping:

#### 1. Iteration-Level Early Stopping (Within Trials)
Finds the optimal number of boosting rounds for each trial:
- **Goal**: Determine best iteration (e.g., iteration 67 out of 200 max)
- **Method**: Stops when no improvement for N consecutive iterations
- **Output**: Each trial uses its optimal iteration count for final predictions

```python
# Configure iteration-level early stopping
iteration_early_stopping=True,      # Enable iteration-level stopping
iteration_patience=15,               # Stop after 15 iterations without improvement  
iteration_min_rounds=30,             # Minimum 30 iterations before stopping
```

#### 2. Trial-Level Early Stopping (Between Trials)
Uses Ax's native strategies to stop underperforming trials:

**Percentile Strategy**: Stops trials in bottom X% compared to others at same iteration
```python
trial_early_stopping_config = {
    "percentile_threshold": 25.0,   # Stop bottom 25% of trials
    "min_progression": 10,          # Wait at least 10 iterations before stopping
    "min_curves": 3,                # Need 3 completed trials before stopping
}
```

**Threshold Strategy**: Stops trials below absolute performance threshold
```python
trial_early_stopping_config = {
    "metric_threshold": 0.85,       # Stop if correlation < 0.85
    "min_progression": 10,          # Check threshold at iteration 10
    "min_curves": 3,                # Need 3 completed trials
}
```

### Metric Calculation
- Uses **weighted correlation** between predictions and targets
- Averages correlation across all folds for each trial
- Finds the best iteration (when early stopping is enabled) by evaluating each iteration

### Artifacts Saved
- `models.pkl`: Best trained models for each fold
- `predictions.pkl`: Complete predictions for the entire dataset
- `iteration_info.pkl`: Best iteration number and parameters

## LightGBM Search Space

The default search space includes:
- `num_leaves`: 10-300
- `learning_rate`: 0.01-0.3  
- `feature_fraction`: 0.4-1.0
- `bagging_fraction`: 0.4-1.0
- `bagging_freq`: 1-7
- `min_child_samples`: 5-100
- `lambda_l1`: 0-10
- `lambda_l2`: 0-10
- `max_depth`: 3-12
- `min_gain_to_split`: 0-15

## Early Stopping Configuration Guide

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `"percentile"` | Stops trials in bottom X percentile | Relative comparison between trials |
| `"threshold"` | Stops trials below absolute threshold | Known performance requirements |

### Configuration Parameters

#### Percentile Strategy Parameters
```python
early_stopping_config = {
    # Core stopping criteria
    "percentile_threshold": 25.0,     # Stop bottom 25% (range: 0-100)
    
    # Progression control  
    "min_progression": 5,             # Min iterations before stopping
    "max_progression": None,          # Max iterations to consider (None = no limit)
    "normalize_progressions": True,   # Normalize iterations to [0,1]
    
    # Safety controls
    "min_curves": 3,                  # Min completed trials before stopping
    "trial_indices_to_ignore": None,  # Specific trials to never stop
}
```

#### Threshold Strategy Parameters
```python
early_stopping_config = {
    # Core stopping criteria
    "metric_threshold": 0.8,          # Minimum correlation required
    
    # Progression control
    "min_progression": 5,             # When to check threshold
    "max_progression": None,          # Max iterations to consider
    "normalize_progressions": True,   # Normalize iterations to [0,1]
    
    # Safety controls  
    "min_curves": 3,                  # Min completed trials before stopping
    "trial_indices_to_ignore": None,  # Specific trials to never stop
}
```

### Usage Examples

#### Conservative Early Stopping (Percentile)
```python
# Stop only very poor trials, wait longer
early_stopping_config = {
    "percentile_threshold": 10.0,  # Only stop bottom 10%
    "min_progression": 10,         # Wait 10 iterations
    "min_curves": 5,               # Need 5 trials before stopping
}
```

#### Aggressive Early Stopping (Percentile)  
```python
# Stop mediocre trials quickly
early_stopping_config = {
    "percentile_threshold": 75.0,  # Stop bottom 75%
    "min_progression": 3,          # Stop after 3 iterations
    "min_curves": 2,               # Start after 2 trials
}
```

#### High-Performance Threshold
```python
# Only continue trials that show promise
early_stopping_config = {
    "metric_threshold": 0.85,      # Need 85% correlation
    "min_progression": 5,          # Check at iteration 5
    "min_curves": 3,
}
```

### Disabling Early Stopping
```python
optimizer = BayesianHyperparameterOptimizer(
    # ... other parameters ...
    use_early_stopping=False,  # Disable early stopping completely
)
```

## Customization

### Custom Search Space
```python
custom_search_space = [
    {
        "name": "num_leaves",
        "type": "range", 
        "bounds": [50, 200],
        "value_type": "int"
    },
    # Add more parameters...
]

optimizer.setup_experiment(df, search_space=custom_search_space)
```

### Configuration Options
- `use_early_stopping`: Enable/disable early stopping
- `max_iterations`: Maximum training iterations per model
- `output_dir`: Directory for saving artifacts

## Testing

Run the test suite:

```bash
python -m pytest test_optimizer.py -v
```

## Example

See `example_lightgbm.py` for a complete working example with synthetic data.

## Extension to Other Models

The pipeline is designed to be easily extensible. To add support for other models:

1. Create a new class inheriting from `BayesianHyperparameterOptimizer`
2. Override `_train_fold_models()` method
3. Define appropriate search space in `_create_search_space()`

## Requirements

- Python 3.11+
- ax-platform>=0.3.0
- lightgbm>=3.3.0
- pandas>=1.5.0
- numpy>=1.20.0
- scikit-learn>=1.0.0

## Notes

- Categorical features are automatically label-encoded
- The pipeline assumes regression problems only
- All models are saved and can be loaded for inference
- Supports both single and multi-objective optimization (extend as needed)