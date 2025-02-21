"""Third layer: Biometric correlation for thought refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np

class BiometricCorrelationLayer(nn.Module):
    """Third layer: Correlate biometric patterns with thought candidates."""
    
    def __init__(
        self,
        freq_dim: int,
        bio_dim: int,
        hidden_dim: int = 128,
        max_candidates: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Biometric feature processing
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Frequency feature processing
        self.freq_encoder = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Correlation attention
        self.correlation_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Candidate scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_candidates)
        )
        
        # Correlation strength estimator
        self.correlation_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        freq_features: torch.Tensor,
        bio_features: torch.Tensor,
        candidate_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through biometric correlation layer.
        
        Args:
            freq_features: Frequency band features
            bio_features: Biometric features
            candidate_mask: Mask of valid candidates from previous layer
            
        Returns:
            Tuple of (correlation_scores, correlation_strength, fused_features)
        """
        # Encode features
        bio_encoded = self.bio_encoder(bio_features)
        freq_encoded = self.freq_encoder(freq_features)
        
        # Apply correlation attention
        corr_features, _ = self.correlation_attention(
            freq_encoded.unsqueeze(1),
            bio_encoded.unsqueeze(1),
            bio_encoded.unsqueeze(1)
        )
        
        # Calculate gate values
        gate_input = torch.cat([
            freq_encoded,
            corr_features.squeeze(1)
        ], dim=1)
        gate = self.gate(gate_input)
        
        # Apply gating
        fused = gate * freq_encoded + (1 - gate) * corr_features.squeeze(1)
        
        # Score candidates
        scores = self.scorer(fused)
        
        # Mask invalid candidates
        masked_scores = scores * candidate_mask.float()
        
        # Estimate correlation strength
        correlation_strength = self.correlation_estimator(fused)
        
        return masked_scores, correlation_strength, fused
    
    def reduce_candidates(
        self,
        correlation_scores: torch.Tensor,
        correlation_strength: torch.Tensor,
        max_candidates: int = 5,
        strength_threshold: float = 0.6
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Reduce thought candidates based on biometric correlation.
        
        Args:
            correlation_scores: Scores for each remaining candidate
            correlation_strength: Strength of biometric correlation
            max_candidates: Maximum number of candidates to keep
            strength_threshold: Minimum correlation strength threshold
            
        Returns:
            Tuple of (filtered_scores, selected_indices)
        """
        # Filter by correlation strength
        strength_mask = correlation_strength > strength_threshold
        
        # Get top-k correlated candidates
        top_scores, top_indices = torch.topk(
            correlation_scores * strength_mask.float(),
            k=min(max_candidates, correlation_scores.shape[1]),
            dim=1
        )
        
        # Create mask for selected candidates
        selected_mask = torch.zeros_like(
            correlation_scores,
            dtype=torch.bool
        )
        selected_mask.scatter_(1, top_indices, True)
        
        return correlation_scores * selected_mask.float(), top_indices
    
    def analyze_correlation_patterns(
        self,
        bio_features: torch.Tensor,
        freq_features: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze correlation patterns between biometric and frequency data."""
        with torch.no_grad():
            # Encode features
            bio_encoded = self.bio_encoder(bio_features)
            freq_encoded = self.freq_encoder(freq_features)
            
            # Calculate correlation matrix
            correlation_matrix = F.cosine_similarity(
                bio_encoded.unsqueeze(1),
                freq_encoded.unsqueeze(0),
                dim=2
            )
            
            # Analyze patterns
            patterns = {
                'mean_correlation': correlation_matrix.mean().item(),
                'max_correlation': correlation_matrix.max().item(),
                'min_correlation': correlation_matrix.min().item(),
                'std_correlation': correlation_matrix.std().item()
            }
            
            return patterns