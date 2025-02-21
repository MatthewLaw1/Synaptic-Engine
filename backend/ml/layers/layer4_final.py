"""Final classification layer for ultimate thought state determination.

This layer makes the final decision on thought states by integrating information
from all previous layers and temporal context. It provides detailed explanations
of its decision-making process and confidence estimates.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, TypeVar, Final
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Type definitions
Tensor = TypeVar('Tensor', bound=torch.Tensor)

# Constants
MIN_THOUGHTS: Final[int] = 1
MAX_THOUGHTS: Final[int] = 5
MIN_CONFIDENCE: Final[float] = 0.0
MAX_CONFIDENCE: Final[float] = 1.0

@dataclass
class DecisionConfig:
    """Configuration for final classification layer."""
    
    feature_dim: int
    hidden_dim: int = 128
    max_thoughts: int = 3
    num_layers: int = 2
    dropout: float = 0.2
    activation: str = 'relu'
    layer_norm: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not MIN_THOUGHTS <= self.max_thoughts <= MAX_THOUGHTS:
            raise ValueError(
                f"max_thoughts must be between {MIN_THOUGHTS} and {MAX_THOUGHTS}"
            )

class TemporalProcessor(nn.Module):
    """Process temporal context for decision making."""
    
    def __init__(self, config: DecisionConfig) -> None:
        """Initialize temporal processor.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=config.hidden_dim * 2,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.layer_norm else None
    
    def forward(
        self,
        features: Tensor,
        hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Process temporal features.
        
        Args:
            features: Input features
            hidden: Optional hidden state
            
        Returns:
            Tuple of (processed_features, new_hidden_state)
        """
        # Process with GRU
        output, new_hidden = self.gru(features, hidden)
        
        # Apply layer norm if enabled
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        
        return output, new_hidden

class DecisionMaker(nn.Module):
    """Make final thought state decisions."""
    
    def __init__(self, config: DecisionConfig) -> None:
        """Initialize decision maker.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        
        self.feature_integrator = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2) if config.layer_norm else nn.Identity(),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout)
        )
        
        self.decision_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.max_thoughts)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {name}")
    
    def forward(
        self,
        features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Make decision from integrated features.
        
        Args:
            features: Integrated features
            
        Returns:
            Tuple of (thought_scores, confidence_scores)
        """
        # Integrate features
        integrated = self.feature_integrator(features)
        
        # Generate thought scores
        scores = self.decision_network(integrated)
        
        # Estimate confidence
        confidence = self.confidence_estimator(integrated)
        
        return scores, confidence

class FinalClassificationLayer(nn.Module):
    """Final layer for thought state determination."""
    
    def __init__(self, config: DecisionConfig) -> None:
        """Initialize final classification layer.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        
        self.config = config
        
        # Feature integration
        self.feature_integrator = nn.Sequential(
            nn.Linear(config.feature_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2) if config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout)
        )
        
        # Temporal processing
        self.temporal_processor = TemporalProcessor(config)
        
        # Decision making
        self.decision_maker = DecisionMaker(config)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on configuration."""
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def forward(
        self,
        sentiment_features: Tensor,
        frequency_features: Tensor,
        biometric_features: Tensor,
        temporal_context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through final classification layer.
        
        Args:
            sentiment_features: Features from sentiment layer
            frequency_features: Features from frequency layer
            biometric_features: Features from biometric layer
            temporal_context: Optional temporal context
            
        Returns:
            Tuple of (thought_scores, confidence_scores, hidden_state)
        """
        # Integrate features from all previous layers
        combined = torch.cat([
            sentiment_features,
            frequency_features,
            biometric_features
        ], dim=1)
        
        integrated = self.feature_integrator(combined)
        
        # Process with temporal context if available
        if temporal_context is not None:
            temporal_input = torch.cat([
                temporal_context,
                integrated.unsqueeze(1)
            ], dim=1)
        else:
            temporal_input = integrated.unsqueeze(1)
        
        # Apply temporal processing
        temporal_output, hidden = self.temporal_processor(temporal_input)
        
        # Get final representation
        final_repr = temporal_output[:, -1]
        
        # Generate thought scores and confidence
        thought_scores, confidence = self.decision_maker(final_repr)
        
        return thought_scores, confidence, hidden
    
    def make_final_decision(
        self,
        thought_scores: Tensor,
        confidence_scores: Tensor,
        min_confidence: float = 0.8
    ) -> Tuple[List[int], List[float]]:
        """Make final thought state decision.
        
        Args:
            thought_scores: Scores for each thought possibility
            confidence_scores: Confidence in each score
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (selected_thoughts, confidence_values)
        """
        # Apply confidence threshold
        confidence_mask = confidence_scores > min_confidence
        
        # Get top-k thoughts
        top_scores, top_indices = torch.topk(
            thought_scores * confidence_mask.float(),
            k=min(self.config.max_thoughts, thought_scores.shape[1]),
            dim=1
        )
        
        # Convert to probabilities
        probabilities = F.softmax(top_scores, dim=1)
        
        # Filter by probability threshold
        prob_mask = probabilities > 0.1  # Remove very low probability thoughts
        
        selected_thoughts = []
        selected_confidences = []
        
        for i in range(top_indices.shape[1]):
            if prob_mask[0, i]:
                selected_thoughts.append(top_indices[0, i].item())
                selected_confidences.append(probabilities[0, i].item())
        
        return selected_thoughts, selected_confidences
    
    def explain_decision(
        self,
        sentiment_features: Tensor,
        frequency_features: Tensor,
        biometric_features: Tensor,
        selected_thoughts: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """Generate explanation for final thought selection.
        
        Args:
            sentiment_features: Features from sentiment layer
            frequency_features: Features from frequency layer
            biometric_features: Features from biometric layer
            selected_thoughts: List of selected thought indices
            
        Returns:
            Dictionary containing contribution analysis for each modality
        """
        with torch.no_grad():
            # Calculate feature contributions
            contributions = {}
            
            for thought_idx in selected_thoughts:
                # Calculate contribution scores
                sentiment_contrib = torch.norm(sentiment_features).item()
                frequency_contrib = torch.norm(frequency_features).item()
                biometric_contrib = torch.norm(biometric_features).item()
                
                # Normalize contributions
                total = sentiment_contrib + frequency_contrib + biometric_contrib
                
                contributions[f'thought_{thought_idx}'] = {
                    'sentiment': sentiment_contrib / total,
                    'frequency': frequency_contrib / total,
                    'biometric': biometric_contrib / total,
                    'confidence': self._calculate_confidence(
                        sentiment_features,
                        frequency_features,
                        biometric_features,
                        thought_idx
                    )
                }
            
            return contributions
    
    def _calculate_confidence(
        self,
        sentiment_features: Tensor,
        frequency_features: Tensor,
        biometric_features: Tensor,
        thought_idx: int
    ) -> float:
        """Calculate confidence score for a specific thought.
        
        Args:
            sentiment_features: Features from sentiment layer
            frequency_features: Features from frequency layer
            biometric_features: Features from biometric layer
            thought_idx: Index of thought to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        # Combine features
        combined = torch.cat([
            sentiment_features,
            frequency_features,
            biometric_features
        ], dim=1)
        
        # Get integrated features
        integrated = self.feature_integrator(combined)
        
        # Calculate confidence
        confidence = self.decision_maker.confidence_estimator(integrated)
        
        return confidence.item()

def create_final_layer(**kwargs) -> FinalClassificationLayer:
    """Create final classification layer.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured FinalClassificationLayer instance
    """
    config = DecisionConfig(**kwargs)
    return FinalClassificationLayer(config)