"""Demo script showcasing the Neural Thought Funneling System."""

import torch
import numpy as np
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from layers import create_pipeline
from cuda_setup import create_cuda_manager
from biometric_processing import BiometricProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(
    num_samples: int = 1,
    num_channels: int = 4,  # Alpha, Beta, Delta, Theta
    sequence_length: int = 256
) -> tuple:
    """Generate sample EEG and biometric data for demonstration."""
    # Generate sample EEG data
    eeg_data = np.random.randn(num_samples, num_channels, sequence_length)
    
    # Generate sample biometric data
    bio_data = {
        'hrv': np.random.rand(num_samples, 100),  # RR intervals
        'blood_pressure': (
            np.random.rand(num_samples, 50),  # Systolic
            np.random.rand(num_samples, 50)   # Diastolic
        ),
        'gsr': np.random.rand(num_samples, 100),
        'respiratory': np.random.rand(num_samples, 100)
    }
    
    return eeg_data, bio_data

def main():
    """Run thought funneling demonstration."""
    logger.info("Starting Neural Thought Funneling Demo")
    
    # Initialize CUDA
    cuda_manager = create_cuda_manager(
        enable_tensorrt=True,
        fp16_mode=True
    )
    
    # Create pipeline
    pipeline = create_pipeline(
        eeg_dim=256,  # Matches sequence_length
        bio_dim=64,   # Combined biometric features
        hidden_dim=128,
        max_thoughts=3
    )
    
    # Optimize model with TensorRT
    pipeline = cuda_manager.optimize_model(
        pipeline,
        input_shapes={
            'eeg_features': (1, 4, 256),
            'bio_features': (1, 64)
        }
    )
    
    # Generate sample data
    logger.info("Generating sample data...")
    eeg_data, bio_data = generate_sample_data()
    
    # Process biometric data
    bio_processor = BiometricProcessor()
    bio_features = bio_processor.extract_all_features(
        rr_intervals=bio_data['hrv'][0],
        blood_pressure=bio_data['blood_pressure'],
        gsr_signal=bio_data['gsr'][0],
        resp_signal=bio_data['respiratory'][0]
    )
    
    # Convert to tensors
    eeg_tensor = torch.FloatTensor(eeg_data)
    bio_tensor = torch.FloatTensor(list(bio_features.values())).unsqueeze(0)
    
    # Run inference
    logger.info("Running thought funneling pipeline...")
    results = cuda_manager.inference(
        pipeline,
        {
            'eeg_features': eeg_tensor,
            'bio_features': bio_tensor
        }
    )
    
    # Display results
    logger.info("\n=== Pipeline Results ===")
    
    # Layer 1: Sentiment Analysis
    logger.info("\nLayer 1 - Sentiment Analysis:")
    valence = results['sentiment']['valence'].item()
    arousal = results['sentiment']['arousal'].item()
    logger.info(f"Valence: {valence:.3f}")
    logger.info(f"Arousal: {arousal:.3f}")
    logger.info(f"Emotional State: {get_emotional_state(valence, arousal)}")
    logger.info(f"Candidates after Layer 1: {torch.sum(results['sentiment']['mask']).item()}")
    
    # Layer 2: Frequency Analysis
    logger.info("\nLayer 2 - Frequency Analysis:")
    freq_confidence = results['frequency']['confidence'].item()
    logger.info(f"Frequency Pattern Confidence: {freq_confidence:.3f}")
    logger.info(f"Candidates after Layer 2: {torch.sum(results['frequency']['mask']).item()}")
    
    # Layer 3: Biometric Correlation
    logger.info("\nLayer 3 - Biometric Correlation:")
    bio_strength = results['biometric']['strength'].item()
    logger.info(f"Biometric Correlation Strength: {bio_strength:.3f}")
    logger.info(f"Candidates after Layer 3: {len(results['biometric']['indices'])}")
    
    # Layer 4: Final Classification
    logger.info("\nLayer 4 - Final Classification:")
    for thought_idx, conf in zip(
        results['final']['thoughts'],
        results['final']['confidence']
    ):
        logger.info(f"Thought ID: {thought_idx}, Confidence: {conf:.3f}")
    
    # Performance metrics
    logger.info("\n=== Performance Metrics ===")
    metrics = cuda_manager.profile_model(
        pipeline,
        {
            'eeg_features': eeg_tensor,
            'bio_features': bio_tensor
        }
    )
    
    logger.info(f"Average Processing Time: {metrics['mean_ms']:.2f} ms")
    logger.info(f"GPU Memory Used: {metrics['memory_mb']:.2f} MB")

def get_emotional_state(valence: float, arousal: float) -> str:
    """Map valence-arousal to emotional state."""
    if valence >= 0:
        if arousal >= 0:
            return 'excited/happy'
        else:
            return 'relaxed/content'
    else:
        if arousal >= 0:
            return 'angry/stressed'
        else:
            return 'sad/depressed'

if __name__ == '__main__':
    main()