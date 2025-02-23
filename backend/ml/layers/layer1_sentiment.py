"""Sentiment analysis layer for initial thought filtering."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, TypeVar, Final, Any
from dataclasses import dataclass
import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)

Tensor = TypeVar('Tensor', bound=torch.Tensor)

MIN_FEATURE_DIM: Final[int] = 32
MAX_FEATURE_DIM: Final[int] = 1024
MIN_HEADS: Final[int] = 1
MAX_HEADS: Final[int] = 16

class SentimentError(Exception):
    pass

@dataclass
class SentimentConfig:
    input_dim: int
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.2
    activation: str = 'relu'
    layer_norm: bool = True
    residual: bool = True
    attention_cache_size: int = 1000
    
    def __post_init__(self) -> None:
        if not MIN_FEATURE_DIM <= self.input_dim <= MAX_FEATURE_DIM:
            raise ValueError(f"input_dim must be between {MIN_FEATURE_DIM} and {MAX_FEATURE_DIM}")
        if not MIN_HEADS <= self.num_heads <= MAX_HEADS:
            raise ValueError(f"num_heads must be between {MIN_HEADS} and {MAX_HEADS}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

class MultiHeadAttention(nn.Module):
    def __init__(self, config: SentimentConfig) -> None:
        super().__init__()
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.layer_norm else None
        
        self.scale = math.sqrt(self.head_dim)
        self._attention_cache = {}
        self.cache_size = config.attention_cache_size
    
    def _validate_inputs(self, query: Tensor, key: Tensor, value: Tensor) -> None:
        if not all(isinstance(x, torch.Tensor) for x in [query, key, value]):
            raise TypeError("All inputs must be PyTorch tensors")
        if not all(x.device == query.device for x in [key, value]):
            raise ValueError("All tensors must be on the same device")
        if not all(torch.isfinite(x).all() for x in [query, key, value]):
            raise ValueError("Inputs contain invalid values")
    
    def _manage_cache(self) -> None:
        if len(self._attention_cache) > self.cache_size:
            self._attention_cache.clear()
    
    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        use_cache: bool = True
    ) -> Tensor:
        try:
            self._validate_inputs(query, key, value)
            batch_size = query.size(0)
            
            if use_cache:
                cache_key = f"{hash(query.data.tobytes())}-{hash(key.data.tobytes())}"
                if cache_key in self._attention_cache:
                    return self._attention_cache[cache_key]
            
            # Linear projections with shape optimization
            q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim)
            k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim)
            v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim)
            
            # Optimize transpose operations
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention scores with memory efficiency
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Apply softmax and dropout
            attn = F.softmax(scores, dim=-1, dtype=torch.float32)
            attn = self.dropout(attn)
            
            # Compute output efficiently
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
            
            out = self.output_proj(out)
            
            if self.layer_norm is not None:
                out = self.layer_norm(out)
            
            if use_cache:
                self._attention_cache[cache_key] = out
                self._manage_cache()
            
            return out
            
        except Exception as e:
            raise SentimentError(f"Attention computation failed: {str(e)}")

class SentimentLayer(nn.Module):
    def __init__(self, config: SentimentConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Initialize with error handling
        try:
            self._init_layers()
        except Exception as e:
            raise SentimentError(f"Layer initialization failed: {str(e)}")
    
    def _init_layers(self) -> None:
        self.eeg_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim) if self.config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.config.dropout)
        )
        
        self.attention = MultiHeadAttention(self.config)
        
        self.bio_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim // 2),
            nn.LayerNorm(self.config.hidden_dim // 2) if self.config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim) if self.config.layer_norm else nn.Identity(),
            self._get_activation()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim) if self.config.layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(self.config.dropout)
        )
        
        self.valence = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            self._get_activation(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.arousal = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            self._get_activation(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def _validate_inputs(self, eeg_features: Tensor, bio_features: Tensor) -> None:
        if not isinstance(eeg_features, torch.Tensor) or not isinstance(bio_features, torch.Tensor):
            raise TypeError("Inputs must be PyTorch tensors")
        if eeg_features.dim() != 3:
            raise ValueError(f"EEG features must be 3D, got shape {eeg_features.shape}")
        if bio_features.dim() != 2:
            raise ValueError(f"Biometric features must be 2D, got shape {bio_features.shape}")
        if not torch.isfinite(eeg_features).all() or not torch.isfinite(bio_features).all():
            raise ValueError("Inputs contain invalid values")
    
    @torch.cuda.amp.autocast()
    def forward(
        self,
        eeg_features: Tensor,
        bio_features: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        try:
            self._validate_inputs(eeg_features, bio_features)
            
            # Process features with memory optimization
            eeg_encoded = self.eeg_encoder(eeg_features)
            eeg_attended = self.attention(eeg_encoded, eeg_encoded, eeg_encoded)
            bio_encoded = self.bio_encoder(bio_features)
            
            # Efficient temporal pooling and fusion
            fused = self.fusion(
                torch.cat([
                    torch.mean(eeg_attended, dim=1),
                    bio_encoded
                ], dim=1)
            )
            
            # Generate predictions
            valence = self.valence(fused)
            arousal = self.arousal(fused)
            
            return valence, arousal, fused
            
        except Exception as e:
            raise SentimentError(f"Forward pass failed: {str(e)}")
    
    @torch.no_grad()
    def filter_candidates(
        self,
        valence: Tensor,
        arousal: Tensor,
        threshold: float = 0.5
    ) -> Tensor:
        try:
            if not 0 <= threshold <= 1:
                raise ValueError("Threshold must be between 0 and 1")
            
            intensity = torch.sqrt(valence**2 + arousal**2)
            mask = intensity > threshold
            
            logger.debug(f"Filtered {torch.sum(mask).item()} candidates with threshold {threshold}")
            
            return mask
            
        except Exception as e:
            raise SentimentError(f"Candidate filtering failed: {str(e)}")

def create_sentiment_layer(**kwargs) -> SentimentLayer:
    try:
        config = SentimentConfig(**kwargs)
        return SentimentLayer(config)
    except Exception as e:
        raise SentimentError(f"Layer creation failed: {str(e)}")