"""
Neural network models for EEG processing.
"""

import torch
import torch.nn as nn

class EEGEmbeddingCNN(nn.Module):
    """
    CNN model for converting EEG feature vectors to embeddings.
    Maps [batch_size, feature_dim] -> [batch_size, embedding_dim]
    using 1D convolutions to capture spatial relationships in the feature dimension.
    """
    
    def __init__(self, input_dim, embedding_dim=64):
        """
        Initialize the CNN model.
        
        Args:
            input_dim: Dimension of input feature vectors
            embedding_dim: Dimension of output embeddings
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * input_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 1, feature_dim]
            
        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def embed_new_sample(eeg_array, model, scaler, device='cpu'):
    """
    Convert raw EEG data to embedding using the trained model.
    
    Args:
        eeg_array: Raw EEG data of shape (4, num_samples)
        model: Trained EEGEmbeddingCNN model
        scaler: Fitted StandardScaler
        device: torch device to use
        
    Returns:
        numpy array of shape (embedding_dim,)
    """
    from .eeg_processing import get_feature_vector
    
    feat_vec = get_feature_vector(eeg_array)
    feat_vec_scaled = scaler.transform(feat_vec.reshape(1, -1))[0]
    
    model.eval()
    with torch.no_grad():
        t = torch.tensor(feat_vec_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        emb = model(t)
    
    return emb.cpu().numpy().reshape(-1)

class TripletLoss(nn.Module):
    """
    Triplet margin loss for training embeddings.
    Pulls anchor and positive samples together while pushing negative samples apart.
    """
    
    def __init__(self, margin=1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Minimum distance between positive and negative pairs
        """
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Scalar loss value
        """
        return self.triplet_loss(anchor, positive, negative)