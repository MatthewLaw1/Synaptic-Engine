"""Neural network models for thought classification using layered reduction."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from .layers import (
    ThoughtReductionPipeline,
    DEFAULT_PARAMS,
    FREQUENCY_BANDS
)

class ThoughtClassifier(nn.Module):
    """Enhanced thought classifier using layered reduction approach."""
    
    def __init__(
        self,
        eeg_dim: int,
        bio_dim: int,
        hidden_dim: int = 128,
        max_thoughts: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.pipeline = ThoughtReductionPipeline(
            eeg_dim=eeg_dim,
            bio_dim=bio_dim,
            hidden_dim=hidden_dim,
            max_thoughts=max_thoughts,
            device=device
        )
        
        self.device = device
        self.to(device)
    
    def forward(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        temporal_context: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the classifier.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            temporal_context: Optional temporal context
            
        Returns:
            Dictionary containing results from each layer
        """
        return self.pipeline(
            eeg_features,
            bio_features,
            temporal_context
        )
    
    def predict(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        return_explanations: bool = False
    ) -> Dict:
        """
        Predict thoughts from input features.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            return_explanations: Whether to return detailed explanations
            
        Returns:
            Dictionary containing predictions and optional explanations
        """
        # Convert inputs to tensors
        eeg_tensor = torch.FloatTensor(eeg_features).to(self.device)
        bio_tensor = torch.FloatTensor(bio_features).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            results = self(eeg_tensor, bio_tensor)
            
            predictions = {
                'thoughts': results['final']['thoughts'],
                'confidence': results['final']['confidence']
            }
            
            if return_explanations:
                predictions['explanations'] = self.pipeline.explain_pipeline(results)
                predictions['reduction_stats'] = self.pipeline.get_reduction_stats(results)
            
            return predictions
    
    def get_frequency_importance(self, eeg_features: np.ndarray) -> Dict[str, float]:
        """Analyze importance of each frequency band."""
        eeg_tensor = torch.FloatTensor(eeg_features).to(self.device)
        
        with torch.no_grad():
            return self.pipeline.frequency_layer.analyze_band_importance(eeg_tensor)

def create_classifier(
    eeg_dim: int,
    bio_dim: int,
    **kwargs
) -> ThoughtClassifier:
    """Factory function to create a thought classifier."""
    return ThoughtClassifier(
        eeg_dim=eeg_dim,
        bio_dim=bio_dim,
        **kwargs
    )

class ModelTrainer:
    """Trainer for the thought classifier."""
    
    def __init__(
        self,
        model: ThoughtClassifier,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters())
    
    def train_step(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        thought_labels: Tensor
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            thought_labels: True thought labels
            
        Returns:
            Dictionary containing loss metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        results = self.model(eeg_features, bio_features)
        
        # Calculate losses for each layer
        losses = {}
        
        # Sentiment layer loss
        sentiment_pred = torch.cat([
            results['sentiment']['valence'],
            results['sentiment']['arousal']
        ], dim=1)
        losses['sentiment'] = nn.MSELoss()(
            sentiment_pred,
            thought_labels.float()  # Assuming labels contain valence-arousal
        )
        
        # Frequency layer loss
        losses['frequency'] = nn.CrossEntropyLoss()(
            results['frequency']['scores'],
            thought_labels
        )
        
        # Biometric layer loss
        losses['biometric'] = nn.CrossEntropyLoss()(
            results['biometric']['scores'],
            thought_labels
        )
        
        # Final layer loss
        losses['final'] = nn.CrossEntropyLoss()(
            torch.tensor(results['final']['thoughts']).unsqueeze(0),
            thought_labels
        )
        
        # Total loss
        total_loss = sum(losses.values())
        total_loss.backward()
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def validate(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        thought_labels: Tensor
    ) -> Dict[str, float]:
        """
        Perform validation.
        
        Args:
            eeg_features: EEG frequency band features
            bio_features: Biometric features
            thought_labels: True thought labels
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model.predict(
                eeg_features.cpu().numpy(),
                bio_features.cpu().numpy(),
                return_explanations=True
            )
            
            # Calculate accuracy
            correct = sum(
                1 for t in predictions['thoughts']
                if t in thought_labels.cpu().numpy()
            )
            accuracy = correct / len(thought_labels)
            
            return {
                'accuracy': accuracy,
                'reduction_stats': predictions['reduction_stats']
            }