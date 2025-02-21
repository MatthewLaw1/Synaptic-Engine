"""Initial sentiment analysis layer for thought classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class SentimentLayer(nn.Module):
    """First layer: Sentiment analysis to filter initial thought space."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # EEG frequency band attention
        self.band_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Biometric processing
        self.bio_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Valence-arousal prediction
        self.valence = nn.Linear(hidden_dim, 1)
        self.arousal = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        bio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through sentiment layer.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            
        Returns:
            Tuple of (valence, arousal, fused_features)
        """
        # Process EEG features with attention
        eeg_attn, _ = self.band_attention(
            eeg_features, eeg_features, eeg_features
        )
        
        # Process biometric features
        bio_encoded = self.bio_encoder(bio_features)
        
        # Fuse features
        fused = self.fusion(
            torch.cat([eeg_attn.mean(1), bio_encoded], dim=1)
        )
        
        # Predict valence and arousal
        valence = torch.tanh(self.valence(fused))
        arousal = torch.tanh(self.arousal(fused))
        
        return valence, arousal, fused

    def filter_candidates(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Filter thought candidates based on emotional intensity."""
        # Calculate emotional intensity
        intensity = torch.sqrt(valence**2 + arousal**2)
        
        # Create mask for high-intensity emotions
        mask = intensity > threshold
        
        return mask