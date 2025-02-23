"""Enhanced training utilities with feature analysis and curriculum learning."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import optuna
from .models import IterativeModelStack, EnhancedTripletLoss
from .sentiment_analysis import SentimentAnalyzer
from .feature_analysis import FeatureAnalyzer
from .eeg_processing import EEGProcessor
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """Dataset for combined EEG and biometric data."""
    
    def __init__(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        thought_labels: np.ndarray,
        sentiment_labels: Optional[np.ndarray] = None,
        feature_mask: Optional[np.ndarray] = None
    ):
        self.eeg_features = torch.FloatTensor(eeg_features)
        self.bio_features = torch.FloatTensor(bio_features)
        self.thought_labels = torch.LongTensor(thought_labels)
        self.sentiment_labels = (
            torch.FloatTensor(sentiment_labels)
            if sentiment_labels is not None else None
        )
        
        if feature_mask is not None:
            self.eeg_features = self.eeg_features[:, feature_mask]
    
    def __len__(self) -> int:
        return len(self.eeg_features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'eeg': self.eeg_features[idx],
            'bio': self.bio_features[idx],
            'thought': self.thought_labels[idx]
        }
        if self.sentiment_labels is not None:
            item['sentiment'] = self.sentiment_labels[idx]
        return item

class ModelTrainer:
    """Enhanced trainer with feature analysis and advanced training strategies."""
    
    def __init__(
        self,
        model_stack: IterativeModelStack,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        experiment_dir: Optional[str] = None
    ):
        self.model_stack = model_stack
        self.sentiment_analyzer = sentiment_analyzer
        self.device = device
        
        # Set up experiment directory
        if experiment_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_dir = f'experiments/run_{timestamp}'
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Initialize optimizers
        self.stack_optimizer = optim.Adam(model_stack.parameters())
        if sentiment_analyzer is not None:
            self.sentiment_optimizer = optim.Adam(sentiment_analyzer.parameters())
        
        # Loss functions
        self.triplet_loss = EnhancedTripletLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        if sentiment_analyzer is not None:
            self.sentiment_loss = nn.MSELoss()
        
        # Initialize EEG processor
        self.eeg_processor = EEGProcessor()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'feature_importance': []
        }
    
    def analyze_features(
        self,
        train_data: Tuple,
        eeg_feature_names: List[str],
        bio_feature_names: List[str]
    ) -> Dict:
        """Analyze feature importance and patterns."""
        analyzer = FeatureAnalyzer(
            eeg_feature_names=eeg_feature_names,
            bio_feature_names=bio_feature_names,
            device=self.device
        )
        
        # Extract features
        eeg_features, bio_features, labels = train_data
        
        # Run comprehensive analysis
        analysis_dir = os.path.join(self.experiment_dir, 'feature_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        results = analyzer.analyze_all(
            eeg_features=eeg_features,
            bio_features=bio_features,
            labels=labels,
            save_dir=analysis_dir
        )
        
        # Log results
        logger.info("Feature analysis completed. Results saved to %s", analysis_dir)
        
        return results
    
    def select_features(
        self,
        train_data: Tuple,
        feature_names: List[str],
        n_features: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Select most important features based on analysis."""
        analyzer = FeatureAnalyzer(
            eeg_feature_names=[],  # Not needed for selection
            bio_feature_names=[],
            device=self.device
        )
        
        if n_features is None:
            n_features = len(feature_names) // 2
        
        features = np.hstack(train_data[:-1])  # Combine all feature arrays
        labels = train_data[-1]
        
        selected_features, selected_indices = analyzer.get_optimal_feature_subset(
            features, labels, n_features
        )
        
        selected_names = [feature_names[i] for i in selected_indices]
        
        logger.info(
            "Selected %d features out of %d",
            len(selected_names),
            len(feature_names)
        )
        
        return selected_features, selected_names
    
    def train_model_stack(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        feature_analysis_freq: int = 5,
        early_stopping_patience: int = 5,
        save_checkpoints: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model stack with periodic feature analysis."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model_stack.train()
            train_losses, train_correct = [], 0
            total_samples = 0
            
            for batch in train_loader:
                eeg = batch['eeg'].to(self.device)
                bio = batch['bio'].to(self.device)
                labels = batch['thought'].to(self.device)
                
                self.stack_optimizer.zero_grad()
                
                # Forward pass through model stack
                embedding = self.model_stack.eeg_embedder(eeg)
                fused = self.model_stack.fusion_model(embedding, bio)
                logits = self.model_stack.classifier(fused)
                
                # Compute loss
                loss = self.ce_loss(logits, labels)
                
                # Add triplet loss if possible
                if len(eeg) >= 3:
                    anchor = embedding[0].unsqueeze(0)
                    positive = embedding[1].unsqueeze(0)
                    negative = embedding[2].unsqueeze(0)
                    triplet_loss = self.triplet_loss(
                        anchor, positive, negative,
                        labels[:3].unsqueeze(1)
                    )
                    loss += triplet_loss
                
                loss.backward()
                self.stack_optimizer.step()
                
                # Track metrics
                train_losses.append(loss.item())
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                total_samples += len(labels)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader)
            
            # Update history
            train_loss = np.mean(train_losses)
            train_acc = train_correct / total_samples
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Periodic feature analysis
            if epoch % feature_analysis_freq == 0:
                self._analyze_current_features(train_loader)
            
            # Save checkpoint if best model
            if val_loss < best_val_loss and save_checkpoints:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs",
                    epoch + 1
                )
                break
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                f"Val Acc: {val_acc:.4f}"
            )
        
        return self.history
    
    def _validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Perform validation pass."""
        self.model_stack.eval()
        val_losses, val_correct = [], 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(self.device)
                bio = batch['bio'].to(self.device)
                labels = batch['thought'].to(self.device)
                
                embedding = self.model_stack.eeg_embedder(eeg)
                fused = self.model_stack.fusion_model(embedding, bio)
                logits = self.model_stack.classifier(fused)
                
                loss = self.ce_loss(logits, labels)
                val_losses.append(loss.item())
                
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                total_samples += len(labels)
        
        return np.mean(val_losses), val_correct / total_samples
    
    def _analyze_current_features(self, train_loader: DataLoader):
        """Analyze feature importance using current model state."""
        analyzer = FeatureAnalyzer([], [], self.device)
        
        # Get SHAP values for current model
        shap_values = analyzer.analyze_shap_values(
            self.model_stack.eeg_embedder,
            train_loader
        )
        
        self.history['feature_importance'].append(
            np.mean(np.abs(shap_values), axis=0)
        )
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model_stack.state_dict(),
            'optimizer_state_dict': self.stack_optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, checkpoint_path)
        
        logger.info("Saved checkpoint to %s", checkpoint_path)

def prepare_data(
    eeg_features: np.ndarray,
    bio_features: np.ndarray,
    thought_labels: np.ndarray,
    sentiment_labels: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    val_size: float = 0.2
) -> Tuple:
    """Prepare train/val/test splits with preprocessing."""
    # Preprocess EEG features
    processor = EEGProcessor()
    eeg_features = processor.preprocess_signal(eeg_features)
    
    # First split into train+val and test
    splits = train_test_split(
        eeg_features, bio_features, thought_labels,
        test_size=test_size, stratify=thought_labels
    )
    
    if sentiment_labels is not None:
        splits = train_test_split(
            *splits, sentiment_labels,
            test_size=test_size, stratify=thought_labels
        )
    
    # Further split train into train and val
    train_splits = train_test_split(
        *splits[:-2],  # Exclude test sets
        test_size=val_size,
        stratify=splits[1][:len(splits[1])*(1-test_size)]  # Stratify by labels
    )
    
    if sentiment_labels is not None:
        return (
            (train_splits[0], train_splits[2], train_splits[4], train_splits[6]),  # Train
            (train_splits[1], train_splits[3], train_splits[5], train_splits[7]),  # Val
            (splits[-2], splits[-1])  # Test
        )
    else:
        return (
            (train_splits[0], train_splits[2], train_splits[4]),  # Train
            (train_splits[1], train_splits[3], train_splits[5]),  # Val
            (splits[-2], splits[-1])  # Test
        )