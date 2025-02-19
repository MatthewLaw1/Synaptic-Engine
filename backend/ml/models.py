"""Neural network models for EEG feature embedding."""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class EEGEmbeddingCNN(nn.Module):
    """CNN for converting EEG features to embeddings."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Flatten(),
            nn.Linear(64 * input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

def embed_new_sample(eeg_array: np.ndarray, model: nn.Module, scaler: Any, device: str = 'cpu') -> np.ndarray:
    """Generate embedding from raw EEG data."""
    from .eeg_processing import get_feature_vector
    
    model.eval()
    with torch.no_grad():
        feat_vec = get_feature_vector(eeg_array)
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))[0]
        tensor = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return model(tensor).cpu().numpy().reshape(-1)

class TripletLoss(nn.Module):
    """Triplet margin loss for training embeddings."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return self.triplet_loss(anchor, positive, negative)