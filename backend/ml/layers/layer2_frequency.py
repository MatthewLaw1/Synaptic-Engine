"""Second layer: Frequency band analysis for thought pattern identification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class FrequencyBandLayer(nn.Module):
    """Second layer: Analyze frequency band patterns to reduce thought candidates."""
    
    FREQUENCY_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_patterns: int = 50,  # Initial number of thought patterns
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_bands = len(self.FREQUENCY_BANDS)
        
        # Band-specific processing
        self.band_processors = nn.ModuleDict({
            band: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for band in self.FREQUENCY_BANDS.keys()
        })
        
        # Cross-band attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Pattern matching layers
        self.pattern_matcher = nn.Sequential(
            nn.Linear(hidden_dim * self.num_bands, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_patterns)
        )
        
        # Confidence scoring
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_bands, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        frequency_features: torch.Tensor,
        prev_layer_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through frequency analysis layer.
        
        Args:
            frequency_features: Dictionary of frequency band features
            prev_layer_features: Features from previous layer
            
        Returns:
            Tuple of (pattern_scores, confidence_scores, processed_features)
        """
        batch_size = frequency_features.shape[0]
        
        # Process each frequency band
        band_features = []
        for band, processor in self.band_processors.items():
            # Extract band features
            band_features.append(processor(frequency_features))
        
        # Stack band features
        stacked_features = torch.stack(band_features, dim=1)
        
        # Apply cross-band attention
        attended_features, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Combine attended features
        combined = attended_features.reshape(batch_size, -1)
        
        # Match against thought patterns
        pattern_scores = self.pattern_matcher(combined)
        
        # Calculate confidence scores
        confidence = self.confidence_scorer(combined)
        
        return pattern_scores, confidence, combined
    
    def reduce_candidates(
        self,
        pattern_scores: torch.Tensor,
        confidence_scores: torch.Tensor,
        max_candidates: int = 20,
        confidence_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reduce thought candidates based on pattern matching and confidence.
        
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
        frequency_features: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze importance of each frequency band."""
        importances = {}
        
        with torch.no_grad():
            # Process each band separately
            for band, processor in self.band_processors.items():
                # Get band features
                band_output = processor(frequency_features)
                
                # Calculate importance score (using L2 norm)
                importance = torch.norm(band_output, p=2, dim=1).mean().item()
                importances[band] = importance
        
        # Normalize importances
        total = sum(importances.values())
        return {k: v/total for k, v in importances.items()}