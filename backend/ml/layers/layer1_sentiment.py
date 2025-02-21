"""Sentiment analysis layer for initial thought filtering.

This layer performs the initial filtering of thought candidates based on emotional
context derived from EEG and biometric signals. It uses attention mechanisms to
process temporal patterns and outputs valence-arousal predictions.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, TypeVar, Final
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

# Type definitions
Tensor = TypeVar('Tensor', bound=torch.Tensor)

# Constants
MIN_FEATURE_DIM: Final[int] = 32
MAX_FEATURE_DIM: Final[int] = 1024
MIN_HEADS: Final[int] = 1
MAX_HEADS: Final[int] = 16

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis layer."""
    
    input_dim: int
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.2
    activation: str = 'relu'
    layer_norm: bool = True
    residual: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not MIN_FEATURE_DIM <= self.input_dim <= MAX_FEATURE_DIM:
            raise ValueError(
                f"input_dim must be between {MIN_FEATURE_DIM} and {MAX_FEATURE_DIM}"
            )
        if not MIN_HEADS <= self.num_heads <= MAX_HEADS:
            raise ValueError(
                f"num_heads must be between {MIN_HEADS} and {MAX_HEADS}"
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

class MultiHeadAttention(nn.Module):
    """Multi-head attention for temporal EEG features."""
    
    def __init__(self, config: SentimentConfig) -> None:
        """Initialize attention module.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.layer_norm else None
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through attention module.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output tensor
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Project output
        out = self.output_proj(out)
        
        # Apply layer normalization if enabled
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        return out

class SentimentLayer(nn.Module):
    """Neural network for sentiment analysis and initial thought filtering."""
    
    def __init__(self, config: SentimentConfig) -> None:
        """Initialize sentiment layer.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        
        self.config = config
        
        # EEG processing branch
        self.eeg_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout)
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(config)
        
        # Biometric processing branch
        self.bio_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2) if config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            self._get_activation()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout)
        )
        
        # Output heads
        self.valence = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self._get_activation(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.arousal = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self._get_activation(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Tanh()
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
        eeg_features: Tensor,
        bio_features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through sentiment layer.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            
        Returns:
            Tuple of (valence, arousal, fused_features)
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        if eeg_features.dim() != 3:
            raise ValueError(
                f"EEG features must be 3D (batch, time, features), "
                f"got shape {eeg_features.shape}"
            )
        
        # Process EEG features
        eeg_encoded = self.eeg_encoder(eeg_features)
        
        # Apply attention
        eeg_attended = self.attention(
            eeg_encoded, eeg_encoded, eeg_encoded
        )
        
        # Process biometric features
        bio_encoded = self.bio_encoder(bio_features)
        
        # Fuse features
        fused = self.fusion(
            torch.cat([
                eeg_attended.mean(1),  # Pool temporal dimension
                bio_encoded
            ], dim=1)
        )
        
        # Generate predictions
        valence = self.valence(fused)
        arousal = self.arousal(fused)
        
        return valence, arousal, fused
    
    def filter_candidates(
        self,
        valence: Tensor,
        arousal: Tensor,
        threshold: float = 0.5
    ) -> Tensor:
        """Filter thought candidates based on emotional intensity.
        
        Args:
            valence: Valence predictions
            arousal: Arousal predictions
            threshold: Minimum emotional intensity threshold
            
        Returns:
            Boolean mask for valid candidates
        """
        # Calculate emotional intensity
        intensity = torch.sqrt(valence**2 + arousal**2)
        
        # Create mask for high-intensity emotions
        mask = intensity > threshold
        
        logger.debug(
            f"Filtered {torch.sum(mask).item()} candidates "
            f"with threshold {threshold}"
        )
        
        return mask

def create_sentiment_layer(**kwargs) -> SentimentLayer:
    """Create sentiment analysis layer.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured SentimentLayer instance
    """
    config = SentimentConfig(**kwargs)
    return SentimentLayer(config)