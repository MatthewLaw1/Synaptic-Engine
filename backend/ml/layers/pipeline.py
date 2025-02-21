"""Pipeline orchestration for iterative thought state reduction."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from .layer1_sentiment import SentimentLayer
from .layer2_frequency import FrequencyBandLayer
from .layer3_biometric import BiometricCorrelationLayer
from .layer4_final import FinalClassificationLayer
import logging

logger = logging.getLogger(__name__)

class ThoughtReductionPipeline(nn.Module):
    """Pipeline for iterative reduction of thought states."""
    
    def __init__(
        self,
        eeg_dim: int,
        bio_dim: int,
        hidden_dim: int = 128,
        max_thoughts: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        
        # Initialize layers
        self.sentiment_layer = SentimentLayer(
            input_dim=eeg_dim,
            hidden_dim=hidden_dim
        )
        
        self.frequency_layer = FrequencyBandLayer(
            input_dim=eeg_dim,
            hidden_dim=hidden_dim,
            num_patterns=50  # Start with 50 possible patterns
        )
        
        self.biometric_layer = BiometricCorrelationLayer(
            freq_dim=hidden_dim,
            bio_dim=bio_dim,
            hidden_dim=hidden_dim,
            max_candidates=10  # Reduce to 10 candidates
        )
        
        self.final_layer = FinalClassificationLayer(
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            max_thoughts=max_thoughts  # Final 1-3 thoughts
        )
        
        # Move to device
        self.to(device)
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        bio_features: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through entire pipeline.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            temporal_context: Optional temporal context
            
        Returns:
            Dictionary containing results from each layer
        """
        results = {}
        
        # Layer 1: Sentiment Analysis
        logger.info("Processing Layer 1: Sentiment Analysis")
        valence, arousal, sentiment_features = self.sentiment_layer(
            eeg_features, bio_features
        )
        sentiment_mask = self.sentiment_layer.filter_candidates(
            valence, arousal
        )
        results['sentiment'] = {
            'valence': valence,
            'arousal': arousal,
            'features': sentiment_features,
            'mask': sentiment_mask
        }
        
        # Layer 2: Frequency Analysis
        logger.info("Processing Layer 2: Frequency Analysis")
        pattern_scores, freq_confidence, freq_features = self.frequency_layer(
            eeg_features, sentiment_features
        )
        pattern_scores, freq_mask = self.frequency_layer.reduce_candidates(
            pattern_scores,
            freq_confidence,
            max_candidates=20  # Reduce to 20 candidates
        )
        results['frequency'] = {
            'scores': pattern_scores,
            'confidence': freq_confidence,
            'features': freq_features,
            'mask': freq_mask
        }
        
        # Layer 3: Biometric Correlation
        logger.info("Processing Layer 3: Biometric Correlation")
        corr_scores, corr_strength, bio_features = self.biometric_layer(
            freq_features,
            bio_features,
            freq_mask
        )
        filtered_scores, selected_indices = self.biometric_layer.reduce_candidates(
            corr_scores,
            corr_strength,
            max_candidates=5  # Reduce to 5 candidates
        )
        results['biometric'] = {
            'scores': filtered_scores,
            'strength': corr_strength,
            'features': bio_features,
            'indices': selected_indices
        }
        
        # Layer 4: Final Classification
        logger.info("Processing Layer 4: Final Classification")
        thought_scores, confidence, hidden = self.final_layer(
            sentiment_features,
            freq_features,
            bio_features,
            temporal_context
        )
        final_thoughts, final_confidences = self.final_layer.make_final_decision(
            thought_scores,
            confidence
        )
        results['final'] = {
            'thoughts': final_thoughts,
            'confidence': final_confidences,
            'hidden': hidden
        }
        
        return results
    
    def explain_pipeline(
        self,
        results: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict]:
        """Generate comprehensive explanation of pipeline decisions."""
        explanation = {}
        
        # Sentiment layer explanation
        explanation['sentiment'] = {
            'emotional_state': self._get_emotional_state(
                results['sentiment']['valence'],
                results['sentiment']['arousal']
            ),
            'intensity': torch.sqrt(
                results['sentiment']['valence']**2 +
                results['sentiment']['arousal']**2
            ).item()
        }
        
        # Frequency layer explanation
        band_importance = self.frequency_layer.analyze_band_importance(
            results['frequency']['features']
        )
        explanation['frequency'] = {
            'dominant_bands': band_importance,
            'pattern_confidence': results['frequency']['confidence'].item()
        }
        
        # Biometric correlation explanation
        bio_patterns = self.biometric_layer.analyze_correlation_patterns(
            results['biometric']['features'],
            results['frequency']['features']
        )
        explanation['biometric'] = bio_patterns
        
        # Final layer explanation
        if len(results['final']['thoughts']) > 0:
            explanation['final'] = self.final_layer.explain_decision(
                results['sentiment']['features'],
                results['frequency']['features'],
                results['biometric']['features'],
                results['final']['thoughts']
            )
        
        return explanation
    
    @staticmethod
    def _get_emotional_state(valence: torch.Tensor, arousal: torch.Tensor) -> str:
        """Map valence-arousal to emotional state."""
        v = valence.item()
        a = arousal.item()
        
        if v >= 0:
            if a >= 0:
                return 'excited/happy'
            else:
                return 'relaxed/content'
        else:
            if a >= 0:
                return 'angry/stressed'
            else:
                return 'sad/depressed'
    
    def get_reduction_stats(self, results: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Get statistics about candidate reduction through pipeline."""
        return {
            'initial_candidates': 50,  # Starting number
            'after_sentiment': torch.sum(results['sentiment']['mask']).item(),
            'after_frequency': torch.sum(results['frequency']['mask']).item(),
            'after_biometric': len(results['biometric']['indices']),
            'final_thoughts': len(results['final']['thoughts'])
        }