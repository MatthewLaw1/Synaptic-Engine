"""Core pipeline implementation for neural thought funneling system."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, TypeVar, Protocol, runtime_checkable, Any
from dataclasses import dataclass
import numpy as np
from .layer1_sentiment import SentimentLayer
from .layer2_frequency import FrequencyBandLayer
from .layer3_biometric import BiometricCorrelationLayer
from .layer4_final import FinalClassificationLayer
import logging
from pathlib import Path
import json
from functools import lru_cache

logger = logging.getLogger(__name__)

Tensor = TypeVar('Tensor', bound=torch.Tensor)
Features = Dict[str, Tensor]
LayerOutput = Dict[str, Any]

class PipelineError(Exception):
    pass

@runtime_checkable
class Layer(Protocol):
    def forward(self, *args, **kwargs) -> Tuple[Tensor, ...]:
        ...
    
    def reduce_candidates(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        ...

@dataclass
class PipelineConfig:
    eeg_dim: int
    bio_dim: int
    hidden_dim: int = 128
    max_thoughts: int = 3
    dropout: float = 0.2
    enable_residual: bool = True
    layer_norm: bool = True
    attention_heads: int = 4
    cache_size: int = 1000

class ThoughtReductionPipeline(nn.Module):
    def __init__(
        self,
        config: PipelineConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        super().__init__()
        
        if config.eeg_dim <= 0 or config.bio_dim <= 0:
            raise ValueError("Input dimensions must be positive")
        if config.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")
        if config.max_thoughts <= 0:
            raise ValueError("Max thoughts must be positive")
        if config.dropout < 0 or config.dropout >= 1:
            raise ValueError("Dropout must be between 0 and 1")
        
        self.config = config
        self.device = device
        
        self._init_layers()
        self.to(device)
        
        self._feature_cache = {}
        self._result_cache = {}
        
        self.metrics: Dict[str, List[float]] = {
            'sentiment_confidence': [],
            'frequency_accuracy': [],
            'biometric_correlation': [],
            'final_confidence': [],
            'processing_time': []
        }
    
    def _init_layers(self) -> None:
        try:
            self.sentiment_layer = SentimentLayer(
                input_dim=self.config.eeg_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.attention_heads,
                dropout=self.config.dropout
            )
            
            self.frequency_layer = FrequencyBandLayer(
                input_dim=self.config.eeg_dim,
                hidden_dim=self.config.hidden_dim,
                num_patterns=50,
                dropout=self.config.dropout
            )
            
            self.biometric_layer = BiometricCorrelationLayer(
                freq_dim=self.config.hidden_dim,
                bio_dim=self.config.bio_dim,
                hidden_dim=self.config.hidden_dim,
                max_candidates=10,
                dropout=self.config.dropout
            )
            
            self.final_layer = FinalClassificationLayer(
                feature_dim=self.config.hidden_dim,
                hidden_dim=self.config.hidden_dim,
                max_thoughts=self.config.max_thoughts,
                dropout=self.config.dropout
            )
        except Exception as e:
            raise PipelineError(f"Layer initialization failed: {str(e)}")
    
    def _validate_inputs(self, eeg_features: Tensor, bio_features: Tensor) -> None:
        if not torch.is_tensor(eeg_features) or not torch.is_tensor(bio_features):
            raise ValueError("Inputs must be PyTorch tensors")
        
        if eeg_features.device != self.device or bio_features.device != self.device:
            raise ValueError("Inputs must be on the correct device")
            
        if not torch.isfinite(eeg_features).all() or not torch.isfinite(bio_features).all():
            raise ValueError("Inputs contain invalid values")
        
        if eeg_features.dim() != 3:
            raise ValueError(f"EEG features must be 3D, got shape {eeg_features.shape}")
        
        if bio_features.dim() != 2:
            raise ValueError(f"Biometric features must be 2D, got shape {bio_features.shape}")
        
        if eeg_features.shape[0] != bio_features.shape[0]:
            raise ValueError("Batch sizes must match")
    
    def _manage_cache(self) -> None:
        if len(self._feature_cache) > self.config.cache_size:
            self._feature_cache.clear()
        if len(self._result_cache) > self.config.cache_size:
            self._result_cache.clear()
    
    @torch.no_grad()
    def forward(
        self,
        eeg_features: Tensor,
        bio_features: Tensor,
        temporal_context: Optional[Tensor] = None,
        use_cache: bool = True
    ) -> Dict[str, LayerOutput]:
        try:
            self._validate_inputs(eeg_features, bio_features)
            
            if use_cache:
                cache_key = f"{hash(eeg_features.data.tobytes())}-{hash(bio_features.data.tobytes())}"
                if cache_key in self._result_cache:
                    return self._result_cache[cache_key]
            
            results: Dict[str, LayerOutput] = {}
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            valence, arousal, sentiment_features = self.sentiment_layer(eeg_features, bio_features)
            sentiment_mask = self.sentiment_layer.filter_candidates(valence, arousal)
            results['sentiment'] = {
                'valence': valence,
                'arousal': arousal,
                'features': sentiment_features,
                'mask': sentiment_mask
            }
            self.metrics['sentiment_confidence'].append(
                torch.mean(torch.abs(valence) + torch.abs(arousal)).item()
            )
            
            pattern_scores, freq_confidence, freq_features = self.frequency_layer(
                eeg_features, sentiment_features
            )
            pattern_scores, freq_mask = self.frequency_layer.reduce_candidates(
                pattern_scores, freq_confidence, max_candidates=20
            )
            results['frequency'] = {
                'scores': pattern_scores,
                'confidence': freq_confidence,
                'features': freq_features,
                'mask': freq_mask
            }
            self.metrics['frequency_accuracy'].append(freq_confidence.mean().item())
            
            corr_scores, corr_strength, bio_features = self.biometric_layer(
                freq_features, bio_features, freq_mask
            )
            filtered_scores, selected_indices = self.biometric_layer.reduce_candidates(
                corr_scores, corr_strength, max_candidates=5
            )
            results['biometric'] = {
                'scores': filtered_scores,
                'strength': corr_strength,
                'features': bio_features,
                'indices': selected_indices
            }
            self.metrics['biometric_correlation'].append(corr_strength.mean().item())
            
            thought_scores, confidence, hidden = self.final_layer(
                sentiment_features, freq_features, bio_features, temporal_context
            )
            final_thoughts, final_confidences = self.final_layer.make_final_decision(
                thought_scores, confidence
            )
            results['final'] = {
                'thoughts': final_thoughts,
                'confidence': final_confidences,
                'hidden': hidden
            }
            self.metrics['final_confidence'].append(
                torch.tensor(final_confidences).mean().item()
            )
            
            end_time.record()
            torch.cuda.synchronize()
            self.metrics['processing_time'].append(start_time.elapsed_time(end_time))
            
            if use_cache:
                self._result_cache[cache_key] = results
                self._manage_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise PipelineError(f"Pipeline processing failed: {str(e)}")
    
    @lru_cache(maxsize=128)
    def explain_pipeline(self, results: Dict[str, LayerOutput]) -> Dict[str, Dict]:
        try:
            explanation = {}
            
            explanation['sentiment'] = {
                'emotional_state': self._get_emotional_state(
                    results['sentiment']['valence'],
                    results['sentiment']['arousal']
                ),
                'intensity': float(torch.sqrt(
                    results['sentiment']['valence']**2 +
                    results['sentiment']['arousal']**2
                ).item())
            }
            
            band_importance = self.frequency_layer.analyze_band_importance(
                results['frequency']['features']
            )
            explanation['frequency'] = {
                'dominant_bands': band_importance,
                'pattern_confidence': float(results['frequency']['confidence'].item())
            }
            
            bio_patterns = self.biometric_layer.analyze_correlation_patterns(
                results['biometric']['features'],
                results['frequency']['features']
            )
            explanation['biometric'] = bio_patterns
            
            if len(results['final']['thoughts']) > 0:
                explanation['final'] = self.final_layer.explain_decision(
                    results['sentiment']['features'],
                    results['frequency']['features'],
                    results['biometric']['features'],
                    results['final']['thoughts']
                )
            
            return explanation
            
        except Exception as e:
            raise PipelineError(f"Pipeline explanation failed: {str(e)}")
    
    def get_reduction_stats(self, results: Dict[str, LayerOutput]) -> Dict[str, int]:
        try:
            return {
                'initial_candidates': 50,
                'after_sentiment': int(torch.sum(results['sentiment']['mask']).item()),
                'after_frequency': int(torch.sum(results['frequency']['mask']).item()),
                'after_biometric': len(results['biometric']['indices']),
                'final_thoughts': len(results['final']['thoughts'])
            }
        except Exception as e:
            raise PipelineError(f"Stats calculation failed: {str(e)}")
    
    def save_metrics(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            metrics_dict = {
                k: [float(v) for v in vals]
                for k, vals in self.metrics.items()
            }
            with open(path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        except Exception as e:
            raise PipelineError(f"Metrics saving failed: {str(e)}")
    
    @staticmethod
    def _get_emotional_state(valence: Tensor, arousal: Tensor) -> str:
        try:
            v = float(valence.item())
            a = float(arousal.item())
            
            if v >= 0:
                return 'excited/happy' if a >= 0 else 'relaxed/content'
            else:
                return 'angry/stressed' if a >= 0 else 'sad/depressed'
        except Exception as e:
            raise PipelineError(f"Emotional state calculation failed: {str(e)}")

def create_pipeline(eeg_dim: int, bio_dim: int, **kwargs) -> ThoughtReductionPipeline:
    try:
        config = PipelineConfig(eeg_dim=eeg_dim, bio_dim=bio_dim, **kwargs)
        return ThoughtReductionPipeline(config)
    except Exception as e:
        raise PipelineError(f"Pipeline creation failed: {str(e)}")