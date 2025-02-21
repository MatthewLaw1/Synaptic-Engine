"""Enhanced EEG processing utilities with advanced filtering and feature extraction."""

import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch, butter, filtfilt, iirnotch
import pywt
import antropy as ant
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

SAMPLING_RATE = 256
WAVELET = 'db4'
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

class EEGProcessor:
    """Advanced EEG signal processing with multiple filtering options."""
    
    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        notch_freq: float = 50.0,  # Power line frequency
        high_pass_freq: float = 0.5,
        low_pass_freq: float = 45.0,
        wavelet: str = WAVELET
    ):
        self.sampling_rate = sampling_rate
        self.notch_freq = notch_freq
        self.high_pass_freq = high_pass_freq
        self.low_pass_freq = low_pass_freq
        self.wavelet = wavelet
        
        # Pre-compute filter coefficients
        self._init_filters()
    
    def _init_filters(self):
        """Initialize all required filters."""
        nyq = self.sampling_rate * 0.5
        
        # High-pass filter
        self.hp_b, self.hp_a = butter(
            4,
            self.high_pass_freq / nyq,
            btype='high'
        )
        
        # Low-pass filter
        self.lp_b, self.lp_a = butter(
            4,
            self.low_pass_freq / nyq,
            btype='low'
        )
        
        # Notch filter
        q_factor = 30.0
        self.notch_b, self.notch_a = iirnotch(
            self.notch_freq,
            q_factor,
            self.sampling_rate
        )
    
    def apply_high_pass(self, signal: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove DC offset and slow drifts."""
        return filtfilt(self.hp_b, self.hp_a, signal, axis=1)
    
    def apply_low_pass(self, signal: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to remove high-frequency noise."""
        return filtfilt(self.lp_b, self.lp_a, signal, axis=1)
    
    def apply_notch(self, signal: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove power line interference."""
        return filtfilt(self.notch_b, self.notch_a, signal, axis=1)
    
    def preprocess_signal(
        self,
        eeg_array: np.ndarray,
        apply_notch: bool = True
    ) -> np.ndarray:
        """Apply full preprocessing pipeline to EEG signal."""
        # Apply high-pass first to remove DC offset
        signal = self.apply_high_pass(eeg_array)
        
        # Apply notch filter if requested
        if apply_notch:
            signal = self.apply_notch(signal)
        
        # Apply low-pass last
        signal = self.apply_low_pass(signal)
        
        return signal
    
    def check_signal_quality(
        self,
        signal: np.ndarray,
        threshold_std: float = 2.0,
        threshold_range: float = 100.0
    ) -> Tuple[bool, Dict[str, float]]:
        """Check signal quality and return metrics."""
        metrics = {
            'std': np.std(signal),
            'range': np.ptp(signal),
            'noise_ratio': self._estimate_noise_ratio(signal)
        }
        
        is_good = (
            metrics['std'] < threshold_std and
            metrics['range'] < threshold_range and
            metrics['noise_ratio'] < 0.5
        )
        
        return is_good, metrics
    
    def _estimate_noise_ratio(self, signal: np.ndarray) -> float:
        """Estimate signal-to-noise ratio using wavelet decomposition."""
        coeffs = pywt.wavedec(signal, self.wavelet, level=4)
        detail_power = np.sum([np.sum(np.square(c)) for c in coeffs[1:]])
        total_power = np.sum(np.square(signal))
        return detail_power / total_power if total_power > 0 else 1.0

def extract_band_power(
    signal: np.ndarray,
    sfreq: int,
    band: Tuple[float, float]
) -> float:
    """Extract power in specific frequency band."""
    freqs, psd = welch(signal, fs=sfreq, nperseg=sfreq*2)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx])

def compute_connectivity(signals: np.ndarray) -> np.ndarray:
    """Compute connectivity matrix between channels."""
    n_channels = signals.shape[0]
    connectivity = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            correlation = np.corrcoef(signals[i], signals[j])[0, 1]
            connectivity[i, j] = correlation
            
    return connectivity

def extract_entropy_features(signal: np.ndarray) -> Dict[str, float]:
    """Extract various entropy measures from signal."""
    return {
        'sample_entropy': ant.sample_entropy(signal),
        'app_entropy': ant.app_entropy(signal),
        'perm_entropy': ant.perm_entropy(signal),
        'spectral_entropy': ant.spectral_entropy(signal, SAMPLING_RATE),
        'svd_entropy': ant.svd_entropy(signal)
    }

def extract_hjorth_parameters(signal: np.ndarray) -> Dict[str, float]:
    """Calculate Hjorth parameters: Activity, Mobility, and Complexity."""
    diff_first = np.diff(signal)
    diff_second = np.diff(diff_first)
    
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff_first) / activity)
    complexity = np.sqrt(np.var(diff_second) / np.var(diff_first)) / mobility
    
    return {
        'activity': activity,
        'mobility': mobility,
        'complexity': complexity
    }

def extract_features(
    eeg_array: np.ndarray,
    sfreq: int = SAMPLING_RATE,
    wavelet: str = WAVELET,
    return_feature_names: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Extract comprehensive feature vector using multiple methods."""
    # Initialize EEG processor
    processor = EEGProcessor(sampling_rate=sfreq)
    
    # Preprocess signal
    filtered = processor.preprocess_signal(eeg_array)
    
    features = []
    feature_names = []
    
    for channel_idx, channel in enumerate(filtered):
        channel_features = {}
        
        # FFT features
        fft_vals = np.abs(fft(channel))
        fft_features = fft_vals[: len(channel)//2]
        features.append(fft_features)
        if return_feature_names:
            feature_names.extend([
                f'ch{channel_idx}_fft_{i}'
                for i in range(len(fft_features))
            ])
        
        # Band power features
        for band_name, band_range in FREQUENCY_BANDS.items():
            power = extract_band_power(channel, sfreq, band_range)
            channel_features[f'power_{band_name}'] = power
        
        # PSD features
        freqs, psd = welch(channel, fs=sfreq, nperseg=sfreq//2)
        features.append(psd)
        if return_feature_names:
            feature_names.extend([
                f'ch{channel_idx}_psd_{i}'
                for i in range(len(psd))
            ])
        
        # Wavelet features
        coeffs = pywt.wavedec(channel, wavelet=wavelet, level=4)
        wavelet_feats = [np.mean(np.abs(c)) for c in coeffs]
        features.append(np.array(wavelet_feats))
        if return_feature_names:
            feature_names.extend([
                f'ch{channel_idx}_wavelet_{i}'
                for i in range(len(wavelet_feats))
            ])
        
        # Entropy features
        entropy_feats = extract_entropy_features(channel)
        channel_features.update({
            f'entropy_{k}': v for k, v in entropy_feats.items()
        })
        
        # Hjorth parameters
        hjorth_feats = extract_hjorth_parameters(channel)
        channel_features.update({
            f'hjorth_{k}': v for k, v in hjorth_feats.items()
        })
        
        # Statistical features
        channel_features.update({
            'mean': np.mean(channel),
            'std': np.std(channel),
            'kurtosis': np.mean((channel - np.mean(channel))**4) / (np.std(channel)**4),
            'skewness': np.mean((channel - np.mean(channel))**3) / (np.std(channel)**3),
            'zero_crossings': np.sum(np.diff(np.signbit(channel)).astype(int))
        })
        
        features.append(np.array(list(channel_features.values())))
        if return_feature_names:
            feature_names.extend(list(channel_features.keys()))
    
    # Add connectivity features
    connectivity = compute_connectivity(filtered)
    features.append(connectivity.flatten())
    if return_feature_names:
        n_channels = len(filtered)
        feature_names.extend([
            f'connectivity_{i}_{j}'
            for i in range(n_channels)
            for j in range(n_channels)
        ])
    
    feature_vector = np.hstack(features)
    
    if return_feature_names:
        return feature_vector, feature_names
    return feature_vector

# Alias for backward compatibility
get_feature_vector = extract_features