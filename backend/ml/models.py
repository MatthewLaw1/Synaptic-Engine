"""Neural network models for signal classification using layered reduction."""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from .layers import (
    ThoughtReductionPipeline as SignalReductionPipeline,
    DEFAULT_PARAMS,
    FREQUENCY_BANDS
)
import logging

logger = logging.getLogger(__name__)

class ModelError(Exception):
    pass

class NeuralClassifier(nn.Module):
    def __init__(
        self,
        eeg_dim: int,
        bio_dim: int,
        hidden_dim: int = 128,
        max_thoughts: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        if eeg_dim <= 0 or bio_dim <= 0:
            raise ValueError("Input dimensions must be positive")
        if hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")
        if max_thoughts <= 0:
            raise ValueError("Max thoughts must be positive")
            
        self.pipeline = SignalReductionPipeline(
            eeg_dim=eeg_dim,
            bio_dim=bio_dim,
            hidden_dim=hidden_dim,
            max_thoughts=max_thoughts,
            device=device
        )
        
        self.device = device
        self._kv_cache = {}
        self.to(device)
    
    def _validate_inputs(self, eeg_features: Tensor, bio_features: Tensor) -> None:
        if not isinstance(eeg_features, Tensor) or not isinstance(bio_features, Tensor):
            raise TypeError("Inputs must be PyTorch tensors")
        if eeg_features.device != self.device or bio_features.device != self.device:
            raise ValueError("Input tensors must be on the correct device")
        if not torch.isfinite(eeg_features).all() or not torch.isfinite(bio_features).all():
            raise ValueError("Input tensors contain invalid values")
    
    def forward(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        temporal_context: Optional[Tensor] = None,
        use_kv_cache: bool = False
    ) -> Dict[str, Tensor]:
        try:
            self._validate_inputs(eeg_features, bio_features)
            
            if use_kv_cache:
                cache_key = f"{hash(eeg_features.data.tobytes())}-{hash(bio_features.data.tobytes())}"
                if cache_key in self._kv_cache:
                    return self._kv_cache[cache_key]
                    
            results = self.pipeline(eeg_features, bio_features, temporal_context)
            
            if use_kv_cache:
                self._kv_cache[cache_key] = results
                if len(self._kv_cache) > 1000:  # Prevent memory leaks
                    self._kv_cache.clear()
                    
            return results
        except Exception as e:
            raise ModelError(f"Forward pass failed: {str(e)}")
    
    def predict(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        return_explanations: bool = False,
        use_vllm: bool = True
    ) -> Dict[str, Any]:
        try:
            if use_vllm:
                if not hasattr(self, '_vllm_engine'):
                    from .vllm_engine import VLLMEngine, VLLMConfig
                    config = VLLMConfig()
                    self._vllm_engine = VLLMEngine(self, config, device=self.device)
                
                return self._vllm_engine.predict(
                    eeg_features,
                    bio_features,
                    return_explanations=return_explanations
                )
            
            eeg_tensor = torch.FloatTensor(eeg_features).to(self.device)
            bio_tensor = torch.FloatTensor(bio_features).to(self.device)
            
            with torch.no_grad():
                results = self(eeg_tensor, bio_tensor, use_kv_cache=True)
                
                predictions = {
                    'classifications': results['final']['classifications'],
                    'confidence_scores': results['final']['confidence']
                }
                
                if return_explanations:
                    predictions['explanations'] = self.pipeline.explain_pipeline(results)
                    predictions['reduction_stats'] = self.pipeline.get_reduction_stats(results)
                
                return predictions
        except Exception as e:
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def get_frequency_importance(self, eeg_features: np.ndarray) -> Dict[str, float]:
        try:
            eeg_tensor = torch.FloatTensor(eeg_features).to(self.device)
            with torch.no_grad():
                return self.pipeline.frequency_layer.analyze_band_importance(eeg_tensor)
        except Exception as e:
            raise ModelError(f"Frequency importance analysis failed: {str(e)}")

def create_classifier(eeg_dim: int, bio_dim: int, **kwargs) -> NeuralClassifier:
    try:
        return NeuralClassifier(eeg_dim=eeg_dim, bio_dim=bio_dim, **kwargs)
    except Exception as e:
        raise ModelError(f"Classifier creation failed: {str(e)}")

class ModelTrainer:
    def __init__(
        self,
        model: NeuralClassifier,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        gradient_clip: float = 1.0
    ):
        self.model = model
        self.device = device
        self.gradient_clip = gradient_clip
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5
        )
    
    def _validate_batch(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        thought_labels: Tensor
    ) -> None:
        if not all(isinstance(x, Tensor) for x in [eeg_features, bio_features, thought_labels]):
            raise TypeError("All inputs must be PyTorch tensors")
        if not all(x.device == self.device for x in [eeg_features, bio_features, thought_labels]):
            raise ValueError("All tensors must be on the correct device")
        if eeg_features.size(0) != bio_features.size(0) != thought_labels.size(0):
            raise ValueError("Batch sizes must match")
    
    def train_step(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        thought_labels: Tensor
    ) -> Dict[str, float]:
        try:
            self._validate_batch(eeg_features, bio_features, thought_labels)
            self.model.train()
            self.optimizer.zero_grad()
            
            results = self.model(eeg_features, bio_features)
            losses = {}
            
            sentiment_pred = torch.cat([
                results['sentiment']['valence'],
                results['sentiment']['arousal']
            ], dim=1)
            losses['sentiment'] = nn.MSELoss()(
                sentiment_pred,
                thought_labels.float()
            )
            
            losses['frequency'] = nn.CrossEntropyLoss()(
                results['frequency']['scores'],
                thought_labels
            )
            
            losses['biometric'] = nn.CrossEntropyLoss()(
                results['biometric']['scores'],
                thought_labels
            )
            
            losses['final'] = nn.CrossEntropyLoss()(
                torch.tensor(results['final']['classifications']).unsqueeze(0),
                thought_labels
            )
            
            total_loss = sum(losses.values())
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            return {k: v.item() for k, v in losses.items()}
        except Exception as e:
            raise ModelError(f"Training step failed: {str(e)}")
    
    def validate(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        thought_labels: Tensor
    ) -> Dict[str, float]:
        try:
            self._validate_batch(eeg_features, bio_features, thought_labels)
            self.model.eval()
            
            with torch.no_grad():
                predictions = self.model.predict(
                    eeg_features.cpu().numpy(),
                    bio_features.cpu().numpy(),
                    return_explanations=True
                )
                
                correct = sum(
                    1 for c in predictions['classifications']
                    if c in thought_labels.cpu().numpy()
                )
                accuracy = correct / len(thought_labels)
                
                metrics = {
                    'accuracy': accuracy,
                    'reduction_stats': predictions['reduction_stats'],
                    'confidence_distribution': predictions.get('confidence_scores', None)
                }
                
                self.scheduler.step(1 - accuracy)  # Update learning rate based on accuracy
                return metrics
        except Exception as e:
            raise ModelError(f"Validation failed: {str(e)}")