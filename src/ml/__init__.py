"""
ML Surrogate Models Module

Machine learning models to replace Cantera simulations with fast predictions.
"""

from .data_generation import TrainingDataGenerator
from .model_training import MLModelTrainer
from .inference import MLPFRPredictor

__all__ = [
    'TrainingDataGenerator',
    'MLModelTrainer',
    'MLPFRPredictor'
]
