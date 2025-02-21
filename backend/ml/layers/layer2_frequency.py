"""Frequency band analysis layer for thought pattern identification.

This layer analyzes EEG frequency bands (alpha, beta, delta, theta) to identify
thought patterns and reduce the candidate space. It uses specialized processing
for each frequency band and combines them through cross-band attention.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, TypeVar, Final
from dataclasses import dataclass
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Type definitions
Tensor = TypeVar('Tensor', bound=torch.Tensor)

class FrequencyBand(Enum):
    """Enumeration of EEG frequency bands."""
    DELTA = auto()  # 0.5-4 Hz
    THETA = auto()  # 4-8 Hz
    ALPHA = auto()  # 8-13 Hz
    BETA = auto()   # 13-30 Hz

# Constants
FREQUENCY_RANGES: Final[Dict[FrequencyBand, Tuple[float, float]]] = {
    FrequencyBand.DELTA: (0.5, 4.0),
    FrequencyBand.THETA: (4.0, 8.0),
    FrequencyBand.ALPHA: (8.0, 13.0),
    FrequencyBand.BETA: (13.0, 30.0)
}

@dataclass
class FrequencyConfig:
    """Configuration for frequency band analysis layer."""
    
    input_dim: int
    hidden_dim: int = 128
    num_patterns: int = 50
    num_heads: int = 4
    dropout: float = 0.2
    activation: str = 'relu'
    layer_norm: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_patterns <= 0:
            raise ValueError("num_patterns must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

class BandProcessor(nn.Module):
    """Processor for individual frequency bands."""
    
    def __init__(
        self,
        band: FrequencyBand,
        config: FrequencyConfig
    ) -> None:
        """Initialize band processor.
        
        Args:
            band: Frequency band to process
            config: Layer configuration
        """
        super().__init__()
        
        self.band = band
        self.config = config
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout)
        )
        
        # Band-specific processing
        self.band_processor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout)
        )
        
        # Pattern detection
        self.pattern_detector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self._get_activation(),
            nn.Linear(config.hidden_dim // 2, config.num_patterns)
        )
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on configuration."""
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Process frequency band features.
        
        Args:
            x: Input features for the frequency band
            
        Returns:
            Tuple of (processed_features, pattern_scores)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply band-specific processing
        processed = self.band_processor(features)
        
        # Detect patterns
        patterns = self.pattern_detector(processed)
        
        return processed, patterns

class CrossBandAttention(nn.Module):
    """Attention mechanism for combining information across frequency bands."""
    
    def __init__(self, config: FrequencyConfig) -> None:
        """Initialize cross-band attention.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.layer_norm else None
    
    def forward(self, features: Dict[FrequencyBand, Tensor]) -> Tensor:
        """Apply cross-band attention.
        
        Args:
            features: Dictionary of features for each frequency band
            
        Returns:
            Combined features across bands
        """
        # Stack features from all bands
        stacked = torch.stack(list(features.values()), dim=1)
        
        # Apply attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Apply layer norm if enabled
        if self.layer_norm is not None:
            attended = self.layer_norm(attended)
        
        return attended

class FrequencyBandLayer(nn.Module):
    """Layer for analyzing frequency band patterns."""
    
    def __init__(self, config: FrequencyConfig) -> None:
        """Initialize frequency band layer.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        
        self.config = config
        
        # Create processors for each frequency band
        self.band_processors = nn.ModuleDict({
            band.name: BandProcessor(band, config)
            for band in FrequencyBand
        })
        
        # Cross-band attention
        self.cross_attention = CrossBandAttention(config)
        
        # Final pattern integration
        self.pattern_integrator = nn.Sequential(
            nn.Linear(config.hidden_dim * len(FrequencyBand), config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2) if config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.num_patterns)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim * len(FrequencyBand), config.hidden_dim),
            self._get_activation(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
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
        frequency_features: Tensor,
        prev_layer_features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through frequency analysis layer.
        
        Args:
            frequency_features: Dictionary of frequency band features
            prev_layer_features: Features from previous layer
            
        Returns:
            Tuple of (pattern_scores, confidence_scores, processed_features)
        """
        batch_size = frequency_features.size(0)
        
        # Process each frequency band
        band_features = {}
        band_patterns = {}
        
        for band in FrequencyBand:
            features, patterns = self.band_processors[band.name](frequency_features)
            band_features[band] = features
            band_patterns[band] = patterns
        
        # Apply cross-band attention
        attended_features = self.cross_attention(band_features)
        
        # Combine features
        combined = attended_features.reshape(batch_size, -1)
        
        # Generate final pattern scores
        pattern_scores = self.pattern_integrator(combined)
        
        # Calculate confidence
        confidence = self.confidence_estimator(combined)
        
        return pattern_scores, confidence, combined
    
    def reduce_candidates(
        self,
        pattern_scores: Tensor,
        confidence_scores: Tensor,
        max_candidates: int = 20,
        confidence_threshold: float = 0.7
    ) -> Tuple[Tensor, Tensor]:
        """Reduce thought candidates based on pattern matching.
        
        Args:
            pattern_scores: Scores for each thought pattern
            confidence_scores: Confidence in each score
            max_candidates: Maximum number of candidates to keep
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (filtered_scores, confidence_mask)
        """
        # Apply confidence threshold
        confidence_mask = confidence_scores > confidence_threshold
        
        # Get top-k patterns
        top_scores, top_indices = torch.topk(
            pattern_scores,
            k=min(max_candidates, pattern_scores.shape[1]),
            dim=1
        )
        
        # Create mask for top patterns
        pattern_mask = torch.zeros_like(pattern_scores, dtype=torch.bool)
        pattern_mask.scatter_(1, top_indices, True)
        
        # Combine masks
        final_mask = confidence_mask & pattern_mask
        
        return pattern_scores * final_mask.float(), final_mask
    
    def analyze_band_importance(
        self,
        frequency_features: Tensor
    ) -> Dict[str, float]:
        """Analyze importance of each frequency band.
        
        Args:
            frequency_features: Input frequency features
            
        Returns:
            Dictionary of importance scores for each band
        """
        importances = {}
        
        with torch.no_grad():
            for band in FrequencyBand:
                processor = self.band_processors[band.name]
                features, _ = processor(frequency_features)
                importance = torch.norm(features, p=2, dim=1).mean().item()
                importances[band.name] = importance
        
        # Normalize importances
        total = sum(importances.values())
        return {k: v/total for k, v in importances.items()}

def create_frequency_layer(**kwargs) -> FrequencyBandLayer:
    """Create frequency band analysis layer.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured FrequencyBandLayer instance
    """
    config = FrequencyConfig(**kwargs)
    return FrequencyBandLayer(config)