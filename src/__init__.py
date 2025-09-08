"""
AI Stock Market Analysis Platform - Source Package
"""

__version__ = "1.0.0"
__author__ = "AI Stock Analysis Team"

# Import main modules for easy access
from . import data_preprocessing
from . import feature_engineering
from . import model_training
from . import evaluation
from . import utils
from . import api_service
from . import security_bot

__all__ = [
    'data_preprocessing',
    'feature_engineering',
    'model_training',
    'evaluation',
    'utils',
    'api_service',
    'security_bot'
]
