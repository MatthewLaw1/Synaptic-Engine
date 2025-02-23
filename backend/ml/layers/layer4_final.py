"""Final classification layer for ultimate thought state determination."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, TypeVar, Final, Any
from dataclasses import dataclass
import logging
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

Tensor = TypeVar('Tensor', bound=torch.Tensor)

class FinalLayerError(Exception):
    pass

MIN_THOUGHTS: Final[int] = 1
MAX_THOUGHTS: Final[int] = 5
MIN_CONFIDENCE: Final[float] = 0.0
MAX_CONFIDENCE: Final[float] = 1.0

@dataclass
class DecisionConfig:
    feature_dim: int
    hidden_dim: int = 128
    max_thoughts: int = 3
    num_layers: int = 2
    dropout: float = 0.2
    activation: str = 'relu'
    layer_norm: bool = True
    cache_size: int = 1000
    
    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not MIN_THOUGHTS <= self.max_thoughts <= MAX_THOUGHTS:
            raise ValueError(f"max_thoughts must be between {MIN_THOUGHTS} and {MAX_THOUGHTS}")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

class TemporalProcessor(nn.Module):
    def __init__(self, config: DecisionConfig) -> None:
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=config.hidden_dim * 2,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.layer_norm else None
        self._temporal_cache = {}
        self.cache_size = config.cache_size
    
    def _validate_inputs(self, features: Tensor, hidden: Optional[Tensor] = None) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a PyTorch tensor")
        if not torch.isfinite(features).all():
            raise ValueError("Features contain invalid values")
        if hidden is not None:
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("Hidden state must be a PyTorch tensor")
            if not torch.isfinite(hidden).all():
                raise ValueError("Hidden state contains invalid values")
    
    def _manage_cache(self) -> None:
        if len(self._temporal_cache) > self.cache_size:
            self._temporal_cache.clear()
    
    @torch.cuda.amp.autocast()
    def forward(
        self,
        features: Tensor,
        hidden: Optional[Tensor] = None,
        use_cache: bool = True
    ) -> Tuple[Tensor, Tensor]:
        try:
            self._validate_inputs(features, hidden)
            
            if use_cache:
                cache_key = f"{hash(features.data.tobytes())}"
                if hidden is not None:
                    cache_key += f"-{hash(hidden.data.tobytes())}"
                if cache_key in self._temporal_cache:
                    return self._temporal_cache[cache_key]
            
            output, new_hidden = self.gru(features, hidden)
            
            if self.layer_norm is not None:
                output = self.layer_norm(output)
            
            result = (output, new_hidden)
            
            if use_cache:
                self._temporal_cache[cache_key] = result
                self._manage_cache()
            
            return result
        except Exception as e:
            raise FinalLayerError(f"Temporal processing failed: {str(e)}")

class DecisionMaker(nn.Module):
    def __init__(self, config: DecisionConfig) -> None:
        super().__init__()
        
        self.feature_integrator = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2) if config.layer_norm else nn.Identity(),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout)
        )
        
        self.decision_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.max_thoughts)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._decision_cache = {}
        self.cache_size = config.cache_size
    
    def _get_activation(self, name: str) -> nn.Module:
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {name}")
    
    def _validate_input(self, features: Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a PyTorch tensor")
        if not torch.isfinite(features).all():
            raise ValueError("Features contain invalid values")
    
    def _manage_cache(self) -> None:
        if len(self._decision_cache) > self.cache_size:
            self._decision_cache.clear()
    
    @torch.cuda.amp.autocast()
    def forward(self, features: Tensor, use_cache: bool = True) -> Tuple[Tensor, Tensor]:
        try:
            self._validate_input(features)
            
            if use_cache:
                cache_key = hash(features.data.tobytes())
                if cache_key in self._decision_cache:
                    return self._decision_cache[cache_key]
            
            integrated = self.feature_integrator(features)
            scores = self.decision_network(integrated)
            confidence = self.confidence_estimator(integrated)
            
            result = (scores, confidence)
            
            if use_cache:
                self._decision_cache[cache_key] = result
                self._manage_cache()
            
            return result
        except Exception as e:
            raise FinalLayerError(f"Decision making failed: {str(e)}")

class FinalClassificationLayer(nn.Module):
    def __init__(self, config: DecisionConfig) -> None:
        super().__init__()
        
        self.config = config
        
        try:
            self._init_layers()
        except Exception as e:
            raise FinalLayerError(f"Layer initialization failed: {str(e)}")
    
    def _init_layers(self) -> None:
        self.feature_integrator = nn.Sequential(
            nn.Linear(self.config.feature_dim * 3, self.config.hidden_dim * 2),
            nn.LayerNorm(self.config.hidden_dim * 2) if self.config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.config.dropout)
        )
        
        self.temporal_processor = TemporalProcessor(self.config)
        self.decision_maker = DecisionMaker(self.config)
    
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def _validate_inputs(
        self,
        sentiment_features: Tensor,
        frequency_features: Tensor,
        biometric_features: Tensor,
        temporal_context: Optional[Tensor] = None
    ) -> None:
        if not all(isinstance(x, torch.Tensor) for x in [sentiment_features, frequency_features, biometric_features]):
            raise TypeError("All features must be PyTorch tensors")
        if not all(torch.isfinite(x).all() for x in [sentiment_features, frequency_features, biometric_features]):
            raise ValueError("Features contain invalid values")
        if temporal_context is not None:
            if not isinstance(temporal_context, torch.Tensor):
                raise TypeError("Temporal context must be a PyTorch tensor")
            if not torch.isfinite(temporal_context).all():
                raise ValueError("Temporal context contains invalid values")
    
    @torch.cuda.amp.autocast()
    def forward(
        self,
        sentiment_features: Tensor,
        frequency_features: Tensor,
        biometric_features: Tensor,
        temporal_context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        try:
            self._validate_inputs(sentiment_features, frequency_features, biometric_features, temporal_context)
            
            combined = torch.cat([sentiment_features, frequency_features, biometric_features], dim=1)
            integrated = self.feature_integrator(combined)
            
            temporal_input = torch.cat([
                temporal_context if temporal_context is not None else integrated.unsqueeze(1),
                integrated.unsqueeze(1)
            ], dim=1)
            
            temporal_output, hidden = self.temporal_processor(temporal_input)
            final_repr = temporal_output[:, -1]
            thought_scores, confidence = self.decision_maker(final_repr)
            
            return thought_scores, confidence, hidden
        except Exception as e:
            raise FinalLayerError(f"Forward pass failed: {str(e)}")
    
    @torch.no_grad()
    def make_final_decision(
        self,
        thought_scores: Tensor,
        confidence_scores: Tensor,
        min_confidence: float = 0.8
    ) -> Tuple[List[int], List[float]]:
        try:
            if not MIN_CONFIDENCE <= min_confidence <= MAX_CONFIDENCE:
                raise ValueError("min_confidence must be between 0 and 1")
            
            confidence_mask = confidence_scores > min_confidence
            
            top_scores, top_indices = torch.topk(
                thought_scores * confidence_mask.float(),
                k=min(self.config.max_thoughts, thought_scores.shape[1]),
                dim=1
            )
            
            probabilities = F.softmax(top_scores, dim=1)
            prob_mask = probabilities > 0.1
            
            selected_thoughts = []
            selected_confidences = []
            
            for i in range(top_indices.shape[1]):
                if prob_mask[0, i]:
                    selected_thoughts.append(int(top_indices[0, i].item()))
                    selected_confidences.append(float(probabilities[0, i].item()))
            
            return selected_thoughts, selected_confidences
        except Exception as e:
            raise FinalLayerError(f"Decision making failed: {str(e)}")
    
    @torch.no_grad()
    @lru_cache(maxsize=128)
    def explain_decision(
        self,
        sentiment_features: Tensor,
        frequency_features: Tensor,
        biometric_features: Tensor,
        selected_thoughts: List[int]
    ) -> Dict[str, Dict[str, float]]:
        try:
            contributions = {}
            
            for thought_idx in selected_thoughts:
                sentiment_contrib = float(torch.norm(sentiment_features).item())
                frequency_contrib = float(torch.norm(frequency_features).item())
                biometric_contrib = float(torch.norm(biometric_features).item())
                
                total = sentiment_contrib + frequency_contrib + biometric_contrib
                
                contributions[f'thought_{thought_idx}'] = {
                    'sentiment': sentiment_contrib / total,
                    'frequency': frequency_contrib / total,
                    'biometric': biometric_contrib / total,
                    'confidence': float(self._calculate_confidence(
                        sentiment_features,
                        frequency_features,
                        biometric_features,
                        thought_idx
                    ))
                }
            
            return contributions
        except Exception as e:
            raise FinalLayerError(f"Decision explanation failed: {str(e)}")
    
    @torch.no_grad()
    def _calculate_confidence(
        self,
        sentiment_features: Tensor,
        frequency_features: Tensor,
        biometric_features: Tensor,
        thought_idx: int
    ) -> float:
        try:
            combined = torch.cat([sentiment_features, frequency_features, biometric_features], dim=1)
            integrated = self.feature_integrator(combined)
            confidence = self.decision_maker.confidence_estimator(integrated)
            return confidence.item()
        except Exception as e:
            raise FinalLayerError(f"Confidence calculation failed: {str(e)}")

def create_final_layer(**kwargs) -> FinalClassificationLayer:
    try:
        config = DecisionConfig(**kwargs)
        return FinalClassificationLayer(config)
    except Exception as e:
        raise FinalLayerError(f"Layer creation failed: {str(e)}")