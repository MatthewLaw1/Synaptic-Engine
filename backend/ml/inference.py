"""Enhanced inference pipeline with feature analysis and advanced processing."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .eeg_processing import EEGProcessor, extract_features as extract_eeg_features
from .biometric_processing import BiometricProcessor
from .sentiment_analysis import SentimentPredictor
from .models import IterativeModelStack
from .feature_analysis import FeatureAnalyzer
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThoughtPrediction:
    """Container for thought prediction results."""
    thought_id: int
    confidence: float
    emotional_state: Optional[str] = None
    sentiment_scores: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    signal_quality: Optional[Dict[str, float]] = None

class ThoughtInferencePipeline:
    """Enhanced pipeline for thought classification with feature analysis."""
    
    def __init__(
        self,
        model_stack: IterativeModelStack,
        sentiment_predictor: Optional[SentimentPredictor] = None,
        thought_labels: Optional[List[str]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        feature_names: Optional[List[str]] = None
    ):
        self.model_stack = model_stack
        self.sentiment_predictor = sentiment_predictor
        self.biometric_processor = BiometricProcessor()
        self.eeg_processor = EEGProcessor()
        self.thought_labels = thought_labels
        self.device = device
        self.feature_names = feature_names
        
        # Set models to evaluation mode
        self.model_stack.eval()
        if self.sentiment_predictor:
            self.sentiment_predictor.model.eval()
        
        # Initialize feature analyzer if feature names are provided
        self.feature_analyzer = (
            FeatureAnalyzer(
                eeg_feature_names=[n for n in feature_names if 'eeg' in n.lower()],
                bio_feature_names=[n for n in feature_names if 'bio' in n.lower()],
                device=device
            )
            if feature_names is not None else None
        )
    
    def preprocess_signals(
        self,
        eeg_data: np.ndarray,
        rr_intervals: Optional[List[float]] = None,
        blood_pressure: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gsr_signal: Optional[np.ndarray] = None,
        resp_signal: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Preprocess EEG and biometric signals with quality checks."""
        # Process EEG signal
        processed_eeg = self.eeg_processor.preprocess_signal(eeg_data)
        
        # Check EEG signal quality
        eeg_quality = {}
        for channel_idx, channel in enumerate(processed_eeg):
            is_good, metrics = self.eeg_processor.check_signal_quality(channel)
            eeg_quality[f'channel_{channel_idx}'] = metrics
            
            if not is_good:
                logger.warning(
                    f"Poor signal quality detected in channel {channel_idx}: {metrics}"
                )
        
        # Extract EEG features
        eeg_features = extract_eeg_features(
            processed_eeg,
            return_feature_names=False
        )
        
        # Extract biometric features
        bio_features = self.biometric_processor.extract_all_features(
            rr_intervals=rr_intervals,
            blood_pressure=blood_pressure,
            gsr_signal=gsr_signal,
            resp_signal=resp_signal
        )
        
        # Convert biometric features dict to array
        bio_array = np.array(list(bio_features.values()))
        
        return eeg_features, bio_array, eeg_quality
    
    def analyze_sentiment(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Analyze emotional state using sentiment predictor."""
        if self.sentiment_predictor is None:
            return None
            
        return self.sentiment_predictor.predict(
            eeg_features,
            bio_features,
            return_emotional_state=True
        )
    
    def analyze_feature_importance(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Analyze feature importance for current prediction."""
        if self.feature_analyzer is None or self.feature_names is None:
            return None
        
        combined_features = np.hstack([eeg_features, bio_features])
        
        shap_values = self.feature_analyzer.analyze_shap_values(
            self.model_stack.eeg_embedder,
            torch.FloatTensor(combined_features).unsqueeze(0)
        )
        
        importance_dict = {
            name: float(abs(value))
            for name, value in zip(self.feature_names, shap_values[0])
        }
        
        return importance_dict
    
    def predict(
        self,
        eeg_data: np.ndarray,
        rr_intervals: Optional[List[float]] = None,
        blood_pressure: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gsr_signal: Optional[np.ndarray] = None,
        resp_signal: Optional[np.ndarray] = None,
        top_k: int = 5,
        sentiment_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        save_analysis: bool = False,
        analysis_dir: Optional[str] = None
    ) -> List[ThoughtPrediction]:

        logger.info("Preprocessing signals...")
        eeg_features, bio_features, signal_quality = self.preprocess_signals(
            eeg_data,
            rr_intervals,
            blood_pressure,
            gsr_signal,
            resp_signal
        )
        
        sentiment_results = None
        if self.sentiment_predictor is not None:
            logger.info("Analyzing sentiment...")
            sentiment_results = self.analyze_sentiment(eeg_features, bio_features)
            
            sentiment_confidence = max(
                abs(sentiment_results['valence']),
                abs(sentiment_results['arousal'])
            )
            if sentiment_confidence < sentiment_threshold:
                logger.warning(
                    f"Low sentiment confidence: {sentiment_confidence:.3f}"
                )
        
        feature_importance = self.analyze_feature_importance(
            eeg_features,
            bio_features
        )
        
        logger.info("Running thought classification...")
        predictions = self.model_stack.predict(
            eeg_features,
            bio_features,
            top_k=top_k
        )
        
        results = []
        for thought_idx, confidence in zip(
            predictions['thought_indices'],
            predictions['confidence_scores']
        ):
            if confidence < confidence_threshold:
                continue
                
            thought_pred = ThoughtPrediction(
                thought_id=thought_idx,
                confidence=confidence,
                emotional_state=sentiment_results.get('emotional_state')
                    if sentiment_results else None,
                sentiment_scores={
                    'valence': sentiment_results['valence'],
                    'arousal': sentiment_results['arousal']
                } if sentiment_results else None,
                feature_importance=feature_importance,
                signal_quality=signal_quality
            )
            results.append(thought_pred)
        
        if save_analysis and analysis_dir:
            self._save_analysis(
                analysis_dir,
                eeg_features,
                bio_features,
                results,
                signal_quality
            )
        
        return results
    
    def _save_analysis(
        self,
        analysis_dir: str,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        predictions: List[ThoughtPrediction],
        signal_quality: Dict[str, float]
    ):
        """Save analysis results to directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(analysis_dir, f'analysis_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        
        if self.feature_analyzer is not None:
            self.feature_analyzer.plot_feature_importance(
                [
                    pred.feature_importance
                    for pred in predictions
                    if pred.feature_importance is not None
                ],
                save_path=os.path.join(save_dir, 'feature_importance.png')
            )
        
        with open(os.path.join(save_dir, 'signal_quality.txt'), 'w') as f:
            for channel, metrics in signal_quality.items():
                f.write(f"{channel}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.3f}\n")
        
        with open(os.path.join(save_dir, 'predictions.txt'), 'w') as f:
            for i, pred in enumerate(predictions):
                f.write(f"Prediction {i+1}:\n")
                f.write(f"  Thought: {self.get_thought_label(pred.thought_id)}\n")
                f.write(f"  Confidence: {pred.confidence:.3f}\n")
                if pred.emotional_state:
                    f.write(f"  Emotional State: {pred.emotional_state}\n")
                if pred.sentiment_scores:
                    f.write("  Sentiment Scores:\n")
                    for k, v in pred.sentiment_scores.items():
                        f.write(f"    {k}: {v:.3f}\n")
    
    def get_thought_label(self, thought_id: int) -> Optional[str]:
        """Get human-readable label for thought ID."""
        if self.thought_labels is None or thought_id >= len(self.thought_labels):
            return None
        return self.thought_labels[thought_id]
    
    def explain_prediction(
        self,
        prediction: ThoughtPrediction
    ) -> Dict[str, str]:
        """Generate detailed explanation of prediction."""
        explanation = {
            'thought': self.get_thought_label(prediction.thought_id)
                or f"Thought {prediction.thought_id}",
            'confidence': f"{prediction.confidence:.2%}"
        }
        
        if prediction.emotional_state:
            explanation['emotional_state'] = prediction.emotional_state
            
        if prediction.sentiment_scores:
            explanation['sentiment'] = (
                f"Valence: {prediction.sentiment_scores['valence']:.2f}, "
                f"Arousal: {prediction.sentiment_scores['arousal']:.2f}"
            )
        
        if prediction.feature_importance:
            top_features = sorted(
                prediction.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            explanation['key_features'] = ", ".join(
                f"{name} ({importance:.3f})"
                for name, importance in top_features
            )
        
        if prediction.signal_quality:
            avg_quality = np.mean([
                metrics['std']
                for metrics in prediction.signal_quality.values()
            ])
            explanation['signal_quality'] = (
                "Good" if avg_quality < 1.5
                else "Fair" if avg_quality < 2.0
                else "Poor"
            )
        
        return explanation

def load_pipeline(
    model_path: str,
    sentiment_model_path: Optional[str] = None,
    thought_labels_path: Optional[str] = None,
    feature_names_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ThoughtInferencePipeline:
    """Load complete inference pipeline from saved models."""
    model_stack = torch.load(model_path, map_location=device)
    
    sentiment_predictor = None
    if sentiment_model_path:
        sentiment_model = torch.load(sentiment_model_path, map_location=device)
        sentiment_predictor = SentimentPredictor(sentiment_model, device)
    
    thought_labels = None
    if thought_labels_path:
        thought_labels = np.load(thought_labels_path, allow_pickle=True)
    
    feature_names = None
    if feature_names_path:
        feature_names = np.load(feature_names_path, allow_pickle=True)
    
    return ThoughtInferencePipeline(
        model_stack=model_stack,
        sentiment_predictor=sentiment_predictor,
        thought_labels=thought_labels,
        device=device,
        feature_names=feature_names
    )