"""Frequency band analysis layer for thought pattern identification."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, TypeVar, Final, Any
from dataclasses import dataclass
import logging
from enum import Enum, auto
from functools import lru_cache

logger = logging.getLogger(__name__)

Tensor = TypeVar('Tensor', bound=torch.Tensor)

class FrequencyError(Exception):
    pass

class FrequencyBand(Enum):
    DELTA = auto()  # 0.5-4 Hz
    THETA = auto()  # 4-8 Hz
    ALPHA = auto()  # 8-13 Hz
    BETA = auto()   # 13-30 Hz

FREQUENCY_RANGES: Final[Dict[FrequencyBand, Tuple[float, float]]] = {
    FrequencyBand.DELTA: (0.5, 4.0),
    FrequencyBand.THETA: (4.0, 8.0),
    FrequencyBand.ALPHA: (8.0, 13.0),
    FrequencyBand.BETA: (13.0, 30.0)
}

@dataclass
class FrequencyConfig:
    input_dim: int
    hidden_dim: int = 128
    num_patterns: int = 50
    num_heads: int = 4
    dropout: float = 0.2
    activation: str = 'relu'
    layer_norm: bool = True
    cache_size: int = 1000
    
    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_patterns <= 0:
            raise ValueError("num_patterns must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

class BandProcessor(nn.Module):
    def __init__(self, band: FrequencyBand, config: FrequencyConfig) -> None:
        super().__init__()
        
        self.band = band
        self.config = config
        self._feature_cache = {}
        
        try:
            self._init_layers()
        except Exception as e:
            raise FrequencyError(f"Band processor initialization failed: {str(e)}")
    
    def _init_layers(self) -> None:
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim) if self.config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.config.dropout)
        )
        
        self.band_processor = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim) if self.config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.config.dropout)
        )
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            self._get_activation(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_patterns)
        )
    
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
        if not torch.isfinite(x).all():
            raise ValueError("Input contains invalid values")
    
    def _manage_cache(self) -> None:
        if len(self._feature_cache) > self.config.cache_size:
            self._feature_cache.clear()
    
    @torch.cuda.amp.autocast()
    def forward(self, x: Tensor, use_cache: bool = True) -> Tuple[Tensor, Tensor]:
        try:
            self._validate_input(x)
            
            if use_cache:
                cache_key = hash(x.data.tobytes())
                if cache_key in self._feature_cache:
                    return self._feature_cache[cache_key]
            
            features = self.feature_extractor(x)
            processed = self.band_processor(features)
            patterns = self.pattern_detector(processed)
            
            result = (processed, patterns)
            
            if use_cache:
                self._feature_cache[cache_key] = result
                self._manage_cache()
            
            return result
        except Exception as e:
            raise FrequencyError(f"Band processing failed: {str(e)}")

class CrossBandAttention(nn.Module):
    def __init__(self, config: FrequencyConfig) -> None:
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.layer_norm else None
        self._attention_cache = {}
        self.cache_size = config.cache_size
    
    def _validate_features(self, features: Dict[FrequencyBand, Tensor]) -> None:
        if not isinstance(features, dict):
            raise TypeError("Features must be a dictionary")
        if not all(isinstance(v, torch.Tensor) for v in features.values()):
            raise TypeError("All features must be PyTorch tensors")
        if not all(torch.isfinite(v).all() for v in features.values()):
            raise ValueError("Features contain invalid values")
    
    def _manage_cache(self) -> None:
        if len(self._attention_cache) > self.cache_size:
            self._attention_cache.clear()
    
    @torch.cuda.amp.autocast()
    def forward(
        self,
        features: Dict[FrequencyBand, Tensor],
        use_cache: bool = True
    ) -> Tensor:
        try:
            self._validate_features(features)
            
            if use_cache:
                cache_key = hash(str([hash(v.data.tobytes()) for v in features.values()]))
                if cache_key in self._attention_cache:
                    return self._attention_cache[cache_key]
            
            stacked = torch.stack(list(features.values()), dim=1)
            attended, _ = self.attention(stacked, stacked, stacked)
            
            if self.layer_norm is not None:
                attended = self.layer_norm(attended)
            
            if use_cache:
                self._attention_cache[cache_key] = attended
                self._manage_cache()
            
            return attended
        except Exception as e:
            raise FrequencyError(f"Cross-band attention failed: {str(e)}")

class FrequencyBandLayer(nn.Module):
    def __init__(self, config: FrequencyConfig) -> None:
        super().__init__()
        
        self.config = config
        
        try:
            self._init_layers()
        except Exception as e:
            raise FrequencyError(f"Layer initialization failed: {str(e)}")
    
    def _init_layers(self) -> None:
        self.band_processors = nn.ModuleDict({
            band.name: BandProcessor(band, self.config)
            for band in FrequencyBand
        })
        
        self.cross_attention = CrossBandAttention(self.config)
        
        self.pattern_integrator = nn.Sequential(
            nn.Linear(self.config.hidden_dim * len(FrequencyBand), self.config.hidden_dim * 2),
            nn.LayerNorm(self.config.hidden_dim * 2) if self.config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim * 2, self.config.num_patterns)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.config.hidden_dim * len(FrequencyBand), self.config.hidden_dim),
            self._get_activation(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def _validate_inputs(
        self,
        frequency_features: Tensor,
        prev_layer_features: Tensor
    ) -> None:
        if not isinstance(frequency_features, torch.Tensor):
            raise TypeError("Frequency features must be a PyTorch tensor")
        if not isinstance(prev_layer_features, torch.Tensor):
            raise TypeError("Previous layer features must be a PyTorch tensor")
        if not torch.isfinite(frequency_features).all():
            raise ValueError("Frequency features contain invalid values")
        if not torch.isfinite(prev_layer_features).all():
            raise ValueError("Previous layer features contain invalid values")
    
    @torch.cuda.amp.autocast()
    def forward(
        self,
        frequency_features: Tensor,
        prev_layer_features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        try:
            self._validate_inputs(frequency_features, prev_layer_features)
            batch_size = frequency_features.size(0)
            
            band_features = {}
            band_patterns = {}
            
            for band in FrequencyBand:
                features, patterns = self.band_processors[band.name](frequency_features)
                band_features[band] = features
                band_patterns[band] = patterns
            
            attended_features = self.cross_attention(band_features)
            combined = attended_features.reshape(batch_size, -1)
            
            pattern_scores = self.pattern_integrator(combined)
            confidence = self.confidence_estimator(combined)
            
            return pattern_scores, confidence, combined
        except Exception as e:
            raise FrequencyError(f"Forward pass failed: {str(e)}")
    
    @torch.no_grad()
    def reduce_candidates(
        self,
        pattern_scores: Tensor,
        confidence_scores: Tensor,
        max_candidates: int = 20,
        confidence_threshold: float = 0.7
    ) -> Tuple[Tensor, Tensor]:
        try:
            if max_candidates <= 0:
                raise ValueError("max_candidates must be positive")
            if not 0 <= confidence_threshold <= 1:
                raise ValueError("confidence_threshold must be between 0 and 1")
            
            confidence_mask = confidence_scores > confidence_threshold
            
            top_scores, top_indices = torch.topk(
                pattern_scores,
                k=min(max_candidates, pattern_scores.shape[1]),
                dim=1
            )
            
            pattern_mask = torch.zeros_like(pattern_scores, dtype=torch.bool)
            pattern_mask.scatter_(1, top_indices, True)
            
            final_mask = confidence_mask & pattern_mask
            
            return pattern_scores * final_mask.float(), final_mask
        except Exception as e:
            raise FrequencyError(f"Candidate reduction failed: {str(e)}")
    
    @torch.no_grad()
    @lru_cache(maxsize=128)
    def analyze_band_importance(self, frequency_features: Tensor) -> Dict[str, float]:
        try:
            importances = {}
            
            for band in FrequencyBand:
                processor = self.band_processors[band.name]
                features, _ = processor(frequency_features)
                importance = float(torch.norm(features, p=2, dim=1).mean().item())
                importances[band.name] = importance
            
            total = sum(importances.values())
            return {k: v/total for k, v in importances.items()}
        except Exception as e:
            raise FrequencyError(f"Band importance analysis failed: {str(e)}")

def create_frequency_layer(**kwargs) -> FrequencyBandLayer:
    try:
        config = FrequencyConfig(**kwargs)
        return FrequencyBandLayer(config)
    except Exception as e:
        raise FrequencyError(f"Layer creation failed: {str(e)}")