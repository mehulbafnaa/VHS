"""
Scripts for training and prediction.
"""

# Import modules to make them accessible from scripts package
from .train import train_model
from .predict import predict_test_set, analyze_predictions

__all__ = [
    'train_model',
    'predict_test_set',
    'analyze_predictions'
]