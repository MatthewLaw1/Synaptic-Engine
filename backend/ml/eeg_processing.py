"""Enhanced EEG processing utilities with advanced filtering and feature extraction."""

import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch, butter, filtfilt, iirnotch
import pywt
import antropy as ant
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SAMPLING_RATE = 256
WAVELET = 'db4'
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

class SignalProcessingError(Exception):
    pass

class EEGProcessor:
    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        notch_freq: float = 50.0,
        high_pass_freq: float = 0.5,
        low_pass_freq: float = 45.0,
        wavelet: str = WAVELET,
        max_workers: int = 4
    ):
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if notch_freq <= 0 or notch_freq >= sampling_rate/2:
            raise ValueError("Invalid notch frequency")
        if high_pass_freq < 0 or high_pass_freq >= sampling_rate/2:
            raise ValueError("Invalid high-pass frequency")
        if low_pass_freq <= high_pass_freq or low_pass_freq >= sampling_rate/2:
            raise ValueError("Invalid low-pass frequency")
        
        self.sampling_rate = sampling_rate
        self.notch_freq = notch_freq
        self.high_pass_freq = high_pass_freq
        self.low_pass_freq = low_pass_freq
        self.wavelet = wavelet
        self.max_workers = max_workers
        
        self._init_filters()
    
    def _init_filters(self) -> None:
        try:
            nyq = self.sampling_rate * 0.5
            self.hp_b, self.hp_a = butter(4, self.high_pass_freq / nyq, btype='high')
            self.lp_b, self.lp_a = butter(4, self.low_pass_freq / nyq, btype='low')
            self.notch_b, self.notch_a = iirnotch(self.notch_freq, 30.0, self.sampling_rate)
        except Exception as e:
            raise SignalProcessingError(f"Filter initialization failed: {str(e)}")
    
    def _validate_signal(self, signal: np.ndarray) -> None:
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy array")
        if signal.size == 0:
            raise ValueError("Signal array is empty")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains invalid values (inf/nan)")
    
    def apply_high_pass(self, signal: np.ndarray) -> np.ndarray:
        self._validate_signal(signal)
        try:
            return filtfilt(self.hp_b, self.hp_a, signal, axis=1)
        except Exception as e:
            raise SignalProcessingError(f"High-pass filtering failed: {str(e)}")
    
    def apply_low_pass(self, signal: np.ndarray) -> np.ndarray:
        self._validate_signal(signal)
        try:
            return filtfilt(self.lp_b, self.lp_a, signal, axis=1)
        except Exception as e:
            raise SignalProcessingError(f"Low-pass filtering failed: {str(e)}")
    
    def apply_notch(self, signal: np.ndarray) -> np.ndarray:
        self._validate_signal(signal)
        try:
            return filtfilt(self.notch_b, self.notch_a, signal, axis=1)
        except Exception as e:
            raise SignalProcessingError(f"Notch filtering failed: {str(e)}")
    
    def preprocess_signal(self, eeg_array: np.ndarray, apply_notch: bool = True) -> np.ndarray:
        try:
            self._validate_signal(eeg_array)
            signal = self.apply_high_pass(eeg_array)
            if apply_notch:
                signal = self.apply_notch(signal)
            signal = self.apply_low_pass(signal)
            return signal
        except Exception as e:
            raise SignalProcessingError(f"Signal preprocessing failed: {str(e)}")
    
    def check_signal_quality(
        self,
        signal: np.ndarray,
        threshold_std: float = 2.0,
        threshold_range: float = 100.0
    ) -> Tuple[bool, Dict[str, float]]:
        self._validate_signal(signal)
        try:
            metrics = {
                'std': float(np.std(signal)),
                'range': float(np.ptp(signal)),
                'noise_ratio': self._estimate_noise_ratio(signal)
            }
            is_good = (
                metrics['std'] < threshold_std and
                metrics['range'] < threshold_range and
                metrics['noise_ratio'] < 0.5
            )
            return is_good, metrics
        except Exception as e:
            raise SignalProcessingError(f"Quality check failed: {str(e)}")
    
    @lru_cache(maxsize=128)
    def _estimate_noise_ratio(self, signal: np.ndarray) -> float:
        try:
            coeffs = pywt.wavedec(signal, self.wavelet, level=4)
            detail_power = np.sum([np.sum(np.square(c)) for c in coeffs[1:]])
            total_power = np.sum(np.square(signal))
            return detail_power / total_power if total_power > 0 else 1.0
        except Exception as e:
            raise SignalProcessingError(f"Noise ratio estimation failed: {str(e)}")

def extract_band_power(signal: np.ndarray, sfreq: int, band: Tuple[float, float]) -> float:
    try:
        freqs, psd = welch(signal, fs=sfreq, nperseg=sfreq*2)
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return float(np.mean(psd[idx]))
    except Exception as e:
        raise SignalProcessingError(f"Band power extraction failed: {str(e)}")

def compute_connectivity(signals: np.ndarray) -> np.ndarray:
    try:
        n_channels = signals.shape[0]
        connectivity = np.zeros((n_channels, n_channels))
        
        if n_channels > 16:
            with ThreadPoolExecutor() as executor:
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        correlation = np.corrcoef(signals[i], signals[j])[0, 1]
                        connectivity[i, j] = correlation
                        connectivity[j, i] = correlation
        else:
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    correlation = np.corrcoef(signals[i], signals[j])[0, 1]
                    connectivity[i, j] = correlation
                    connectivity[j, i] = correlation
        
        return connectivity
    except Exception as e:
        raise SignalProcessingError(f"Connectivity computation failed: {str(e)}")

def extract_entropy_features(signal: np.ndarray) -> Dict[str, float]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return {
                'sample_entropy': float(ant.sample_entropy(signal)),
                'app_entropy': float(ant.app_entropy(signal)),
                'perm_entropy': float(ant.perm_entropy(signal)),
                'spectral_entropy': float(ant.spectral_entropy(signal, SAMPLING_RATE)),
                'svd_entropy': float(ant.svd_entropy(signal))
            }
    except Exception as e:
        raise SignalProcessingError(f"Entropy feature extraction failed: {str(e)}")

def extract_hjorth_parameters(signal: np.ndarray) -> Dict[str, float]:
    try:
        diff_first = np.diff(signal)
        diff_second = np.diff(diff_first)
        
        activity = float(np.var(signal))
        if activity == 0:
            return {'activity': 0.0, 'mobility': 0.0, 'complexity': 0.0}
        
        mobility = float(np.sqrt(np.var(diff_first) / activity))
        var_diff_first = float(np.var(diff_first))
        
        complexity = 0.0 if var_diff_first == 0 else float(np.sqrt(np.var(diff_second) / var_diff_first)) / mobility
        
        return {
            'activity': activity,
            'mobility': mobility,
            'complexity': complexity
        }
    except Exception as e:
        raise SignalProcessingError(f"Hjorth parameter calculation failed: {str(e)}")

def extract_features(
    eeg_array: np.ndarray,
    sfreq: int = SAMPLING_RATE,
    wavelet: str = WAVELET,
    return_feature_names: bool = False,
    parallel: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    try:
        processor = EEGProcessor(sampling_rate=sfreq)
        filtered = processor.preprocess_signal(eeg_array)
        features = []
        feature_names = []
        
        def process_channel(channel_idx: int, channel: np.ndarray) -> Dict[str, Any]:
            channel_features = {}
            
            fft_vals = np.abs(fft(channel))
            fft_features = fft_vals[: len(channel)//2]
            
            for band_name, band_range in FREQUENCY_BANDS.items():
                power = extract_band_power(channel, sfreq, band_range)
                channel_features[f'power_{band_name}'] = power
            
            _, psd = welch(channel, fs=sfreq, nperseg=sfreq//2)
            
            coeffs = pywt.wavedec(channel, wavelet=wavelet, level=4)
            wavelet_feats = [float(np.mean(np.abs(c))) for c in coeffs]
            
            entropy_feats = extract_entropy_features(channel)
            channel_features.update({f'entropy_{k}': v for k, v in entropy_feats.items()})
            
            hjorth_feats = extract_hjorth_parameters(channel)
            channel_features.update({f'hjorth_{k}': v for k, v in hjorth_feats.items()})
            
            channel_features.update({
                'mean': float(np.mean(channel)),
                'std': float(np.std(channel)),
                'kurtosis': float(np.mean((channel - np.mean(channel))**4) / (np.std(channel)**4)),
                'skewness': float(np.mean((channel - np.mean(channel))**3) / (np.std(channel)**3)),
                'zero_crossings': int(np.sum(np.diff(np.signbit(channel)).astype(int)))
            })
            
            return {
                'fft': fft_features,
                'psd': psd,
                'wavelet': wavelet_feats,
                'channel_features': channel_features
            }
        
        if parallel and len(filtered) > 1:
            with ThreadPoolExecutor() as executor:
                channel_results = list(executor.map(
                    lambda x: process_channel(x[0], x[1]),
                    enumerate(filtered)
                ))
        else:
            channel_results = [
                process_channel(idx, channel)
                for idx, channel in enumerate(filtered)
            ]
        
        for channel_idx, result in enumerate(channel_results):
            features.extend([
                result['fft'],
                result['psd'],
                np.array(result['wavelet']),
                np.array(list(result['channel_features'].values()))
            ])
            
            if return_feature_names:
                feature_names.extend([f'ch{channel_idx}_fft_{i}' for i in range(len(result['fft']))])
                feature_names.extend([f'ch{channel_idx}_psd_{i}' for i in range(len(result['psd']))])
                feature_names.extend([f'ch{channel_idx}_wavelet_{i}' for i in range(len(result['wavelet']))])
                feature_names.extend([f'ch{channel_idx}_{k}' for k in result['channel_features'].keys()])
        
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
        return (feature_vector, feature_names) if return_feature_names else feature_vector
        
    except Exception as e:
        raise SignalProcessingError(f"Feature extraction failed: {str(e)}")

get_feature_vector = extract_features