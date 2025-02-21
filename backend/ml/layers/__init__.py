"""Layer modules for thought state reduction pipeline."""

from .layer1_sentiment import SentimentLayer
from .layer2_frequency import FrequencyBandLayer
from .layer3_biometric import BiometricCorrelationLayer
from .layer4_final import FinalClassificationLayer
from .pipeline import ThoughtReductionPipeline

__all__ = [
    'SentimentLayer',
    'FrequencyBandLayer',
    'BiometricCorrelationLayer',
    'FinalClassificationLayer',
    'ThoughtReductionPipeline'
]

# Layer configuration constants
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# Default reduction parameters
DEFAULT_PARAMS = {
    'sentiment_threshold': 0.5,
    'frequency_max_candidates': 20,
    'frequency_confidence_threshold': 0.7,
    'biometric_max_candidates': 5,
    'biometric_strength_threshold': 0.6,
    'final_max_thoughts': 3,
    'final_confidence_threshold': 0.8
}

def create_pipeline(
    eeg_dim: int,
    bio_dim: int,
    hidden_dim: int = 128,
    max_thoughts: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ThoughtReductionPipeline:
    """
    Create a thought reduction pipeline.
    
    Args:
        eeg_dim: Dimension of EEG features
        bio_dim: Dimension of biometric features
        hidden_dim: Hidden layer dimension
        max_thoughts: Maximum number of final thoughts
        device: Device to run the model on
        
    Returns:
        Configured ThoughtReductionPipeline instance
    """
    return ThoughtReductionPipeline(
        eeg_dim=eeg_dim,
        bio_dim=bio_dim,
        hidden_dim=hidden_dim,
        max_thoughts=max_thoughts,
        device=device
    )