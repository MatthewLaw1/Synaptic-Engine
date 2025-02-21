"""Sentiment analysis module for EEG and biometric data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

class SentimentAttention(nn.Module):
    """Multi-head attention for temporal EEG features."""
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "Feature dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

class SentimentAnalyzer(nn.Module):
    """Neural network for sentiment analysis combining EEG and biometric data."""
    
    def __init__(
        self,
        eeg_feature_dim: int,
        bio_feature_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # EEG processing branch
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.eeg_attention_layers = nn.ModuleList([
            SentimentAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Biometric processing branch
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Fusion and output layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads for valence and arousal
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
    def forward(
        self,
        eeg_features: torch.Tensor,
        bio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the sentiment analyzer.
        
        Args:
            eeg_features: Tensor of shape (batch_size, sequence_length, eeg_feature_dim)
            bio_features: Tensor of shape (batch_size, bio_feature_dim)
            
        Returns:
            Tuple of (valence, arousal) predictions
        """
        # Process EEG features
        x_eeg = self.eeg_encoder(eeg_features)
        
        # Apply attention layers
        for attn_layer in self.eeg_attention_layers:
            x_eeg = x_eeg + attn_layer(x_eeg)  # Residual connection
        
        # Global average pooling over time
        x_eeg = x_eeg.mean(dim=1)
        
        # Process biometric features
        x_bio = self.bio_encoder(bio_features)
        
        # Concatenate and fuse features
        x = torch.cat([x_eeg, x_bio], dim=1)
        x = self.fusion_layer(x)
        
        # Generate predictions
        valence = self.valence_head(x)
        arousal = self.arousal_head(x)
        
        return valence, arousal

class SentimentPredictor:
    """Wrapper class for sentiment prediction and analysis."""
    
    def __init__(
        self,
        model: SentimentAnalyzer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def predict(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        return_emotional_state: bool = False
    ) -> Dict[str, float]:
        """
        Predict sentiment from EEG and biometric features.
        
        Args:
            eeg_features: EEG features array
            bio_features: Biometric features array
            return_emotional_state: Whether to return categorical emotional state
            
        Returns:
            Dictionary containing predictions and optional emotional state
        """
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg_features).unsqueeze(0).to(self.device)
        bio_tensor = torch.FloatTensor(bio_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            valence, arousal = self.model(eeg_tensor, bio_tensor)
            
        results = {
            'valence': valence.item(),
            'arousal': arousal.item()
        }
        
        if return_emotional_state:
            results['emotional_state'] = self._get_emotional_state(
                valence.item(), arousal.item()
            )
            
        return results
    
    @staticmethod
    def _get_emotional_state(valence: float, arousal: float) -> str:
        """Map valence and arousal to emotional state."""
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

def create_sentiment_analyzer(
    eeg_feature_dim: int,
    bio_feature_dim: int,
    **kwargs
) -> SentimentAnalyzer:
    """Factory function to create a sentiment analyzer model."""
    return SentimentAnalyzer(
        eeg_feature_dim=eeg_feature_dim,
        bio_feature_dim=bio_feature_dim,
        **kwargs
    )