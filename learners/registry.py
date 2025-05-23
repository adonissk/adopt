"""
Registry module for automatically registering all available learners.
"""

from .factory import LearnerFactory
from .lightgbm_learner import LightGBMLearner
from .catboost_learner import CatBoostLearner
from .random_forest_learner import RandomForestLearner


def register_default_learners():
    """Register all default learners with the factory."""
    
    # Register LightGBM (always available)
    LearnerFactory.register_learner("lightgbm", LightGBMLearner)
    
    # Register CatBoost (if available)
    try:
        import catboost
        LearnerFactory.register_learner("catboost", CatBoostLearner)
    except ImportError:
        print("CatBoost not available - skipping registration")
    
    # Register Random Forest (always available via sklearn)
    LearnerFactory.register_learner("sklearn_random_forest", RandomForestLearner)


# Auto-register learners when module is imported
register_default_learners()