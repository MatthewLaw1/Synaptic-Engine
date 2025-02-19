"""
Machine learning module for EEG processing and thought classification.
"""

from .eeg_processing import get_feature_vector
from .models import EEGEmbeddingCNN, embed_new_sample
from .inference import process_eeg_file, main as process_eeg, CloudStorageManager, storage_manager

__all__ = [
    'get_feature_vector',
    'EEGEmbeddingCNN',
    'embed_new_sample',
    'process_eeg_file',
    'process_eeg',
    'CloudStorageManager',
    'storage_manager'
]