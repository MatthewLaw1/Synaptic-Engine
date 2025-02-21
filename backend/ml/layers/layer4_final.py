"""Final layer: Ultimate thought state classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np

class FinalClassificationLayer(nn.Module):
    """Fourth layer: Final thought state determination."""
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        max_thoughts: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.max_thoughts = max_thoughts
        
        # Feature integration
        self.feature_integrator = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim * 2),  # Combine features from all previous layers
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal context processing
        self.temporal_processor = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Final decision layers
        self.decision_maker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_thoughts)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        sentiment_features: torch.Tensor,
        frequency_features: torch.Tensor,
        biometric_features: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through final classification layer.
        
        Args:
            sentiment_features: Features from sentiment layer
            frequency_features: Features from frequency layer
            biometric_features: Features from biometric layer
            temporal_context: Optional temporal context from previous states
            
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
        thought_scores = self.decision_maker(final_repr)
        confidence = self.confidence_estimator(final_repr)
        
        return thought_scores, confidence, hidden
    
    def make_final_decision(
        self,
        thought_scores: torch.Tensor,
        confidence_scores: torch.Tensor,
        min_confidence: float = 0.8
    ) -> Tuple[List[int], List[float]]:
        """
        Make final thought state decision.
        
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
            k=min(self.max_thoughts, thought_scores.shape[1]),
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
        sentiment_features: torch.Tensor,
        frequency_features: torch.Tensor,
        biometric_features: torch.Tensor,
        selected_thoughts: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate explanation for final thought selection.
        
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
                # Calculate contribution scores using gradient approximation
                sentiment_contrib = torch.norm(sentiment_features).item()
                frequency_contrib = torch.norm(frequency_features).item()
                biometric_contrib = torch.norm(biometric_features).item()
                
                # Normalize contributions
                total = sentiment_contrib + frequency_contrib + biometric_contrib
                
                contributions[f'thought_{thought_idx}'] = {
                    'sentiment': sentiment_contrib / total,
                    'frequency': frequency_contrib / total,
                    'biometric': biometric_contrib / total
                }
            
            return contributions