"""Core pipeline implementation for neural thought funneling system.

This module implements the main pipeline that orchestrates the flow between
different layers of the thought classification system. It handles data validation,
error checking, and ensures proper state reduction at each stage.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, TypeVar, Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
from .layer1_sentiment import SentimentLayer
from .layer2_frequency import FrequencyBandLayer
from .layer3_biometric import BiometricCorrelationLayer
from .layer4_final import FinalClassificationLayer
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Type definitions
Tensor = TypeVar('Tensor', bound=torch.Tensor)
Features = Dict[str, Tensor]
LayerOutput = Dict[str, Tensor]

@runtime_checkable
class Layer(Protocol):
    """Protocol defining the interface for pipeline layers."""
    
    def forward(self, *args, **kwargs) -> Tuple[Tensor, ...]:
        """Forward pass through the layer."""
        ...
    
    def reduce_candidates(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Reduce the number of candidates."""
        ...

@dataclass
class PipelineConfig:
    """Configuration for the thought reduction pipeline."""
    
    eeg_dim: int
    bio_dim: int
    hidden_dim: int = 128
    max_thoughts: int = 3
    dropout: float = 0.2
    enable_residual: bool = True
    layer_norm: bool = True
    attention_heads: int = 4

class ThoughtReductionPipeline(nn.Module):
    """Pipeline for iterative reduction of thought states.
    
    This pipeline implements a multi-stage process for reducing possible thought
    states through sentiment analysis, frequency band processing, biometric
    correlation, and final classification.
    
    Attributes:
        config: Pipeline configuration
        device: Device to run computations on
        sentiment_layer: Initial sentiment analysis layer
        frequency_layer: Frequency band analysis layer
        biometric_layer: Biometric correlation layer
        final_layer: Final classification layer
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            device: Device to run computations on
        """
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Initialize layers
        self._init_layers()
        
        # Move to device
        self.to(device)
        
        # Initialize metrics tracking
        self.metrics: Dict[str, List[float]] = {
            'sentiment_confidence': [],
            'frequency_accuracy': [],
            'biometric_correlation': [],
            'final_confidence': []
        }
    
    def _init_layers(self) -> None:
        """Initialize all pipeline layers."""
        self.sentiment_layer = SentimentLayer(
            input_dim=self.config.eeg_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout
        )
        
        self.frequency_layer = FrequencyBandLayer(
            input_dim=self.config.eeg_dim,
            hidden_dim=self.config.hidden_dim,
            num_patterns=50,  # Initial patterns
            dropout=self.config.dropout
        )
        
        self.biometric_layer = BiometricCorrelationLayer(
            freq_dim=self.config.hidden_dim,
            bio_dim=self.config.bio_dim,
            hidden_dim=self.config.hidden_dim,
            max_candidates=10,
            dropout=self.config.dropout
        )
        
        self.final_layer = FinalClassificationLayer(
            feature_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            max_thoughts=self.config.max_thoughts,
            dropout=self.config.dropout
        )
    
    def _validate_inputs(
        self,
        eeg_features: Tensor,
        bio_features: Tensor
    ) -> None:
        """Validate input tensors.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not torch.is_tensor(eeg_features) or not torch.is_tensor(bio_features):
            raise ValueError("Inputs must be PyTorch tensors")
        
        if eeg_features.dim() != 3:
            raise ValueError(
                f"EEG features must be 3D (batch, channels, time), "
                f"got shape {eeg_features.shape}"
            )
        
        if bio_features.dim() != 2:
            raise ValueError(
                f"Biometric features must be 2D (batch, features), "
                f"got shape {bio_features.shape}"
            )
        
        if eeg_features.shape[0] != bio_features.shape[0]:
            raise ValueError("Batch sizes must match between EEG and biometric features")
    
    def forward(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        temporal_context: Optional[Tensor] = None
    ) -> Dict[str, LayerOutput]:
        """Forward pass through the pipeline.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            temporal_context: Optional temporal context
            
        Returns:
            Dictionary containing results from each layer
        
        Raises:
            ValueError: If inputs are invalid
        """
        self._validate_inputs(eeg_features, bio_features)
        results: Dict[str, LayerOutput] = {}
        
        try:
            # Layer 1: Sentiment Analysis
            logger.debug("Processing Layer 1: Sentiment Analysis")
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
            self.metrics['sentiment_confidence'].append(
                torch.mean(torch.abs(valence) + torch.abs(arousal)).item()
            )
            
            # Layer 2: Frequency Analysis
            logger.debug("Processing Layer 2: Frequency Analysis")
            pattern_scores, freq_confidence, freq_features = self.frequency_layer(
                eeg_features, sentiment_features
            )
            pattern_scores, freq_mask = self.frequency_layer.reduce_candidates(
                pattern_scores,
                freq_confidence,
                max_candidates=20
            )
            results['frequency'] = {
                'scores': pattern_scores,
                'confidence': freq_confidence,
                'features': freq_features,
                'mask': freq_mask
            }
            self.metrics['frequency_accuracy'].append(freq_confidence.mean().item())
            
            # Layer 3: Biometric Correlation
            logger.debug("Processing Layer 3: Biometric Correlation")
            corr_scores, corr_strength, bio_features = self.biometric_layer(
                freq_features,
                bio_features,
                freq_mask
            )
            filtered_scores, selected_indices = self.biometric_layer.reduce_candidates(
                corr_scores,
                corr_strength,
                max_candidates=5
            )
            results['biometric'] = {
                'scores': filtered_scores,
                'strength': corr_strength,
                'features': bio_features,
                'indices': selected_indices
            }
            self.metrics['biometric_correlation'].append(corr_strength.mean().item())
            
            # Layer 4: Final Classification
            logger.debug("Processing Layer 4: Final Classification")
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
            self.metrics['final_confidence'].append(
                torch.tensor(final_confidences).mean().item()
            )
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")
            raise
        
        return results
    
    def explain_pipeline(
        self,
        results: Dict[str, LayerOutput]
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
    
    def get_reduction_stats(self, results: Dict[str, LayerOutput]) -> Dict[str, int]:
        """Get statistics about candidate reduction through pipeline."""
        return {
            'initial_candidates': 50,
            'after_sentiment': torch.sum(results['sentiment']['mask']).item(),
            'after_frequency': torch.sum(results['frequency']['mask']).item(),
            'after_biometric': len(results['biometric']['indices']),
            'final_thoughts': len(results['final']['thoughts'])
        }
    
    def save_metrics(self, path: Path) -> None:
        """Save pipeline metrics to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    @staticmethod
    def _get_emotional_state(valence: Tensor, arousal: Tensor) -> str:
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

def create_pipeline(
    eeg_dim: int,
    bio_dim: int,
    **kwargs
) -> ThoughtReductionPipeline:
    """Create a thought reduction pipeline.
    
    Args:
        eeg_dim: Dimension of EEG features
        bio_dim: Dimension of biometric features
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ThoughtReductionPipeline instance
    """
    config = PipelineConfig(
        eeg_dim=eeg_dim,
        bio_dim=bio_dim,
        **kwargs
    )
    return ThoughtReductionPipeline(config)