"""
Core EEG processing utilities for feature extraction and embedding generation.
"""

import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch
import pywt

SAMPLING_RATE = 256
WAVELET = 'db4'

def bandpass_filter_dummy(eeg_array, low=0.5, high=40, sfreq=256):
    """
    Placeholder for bandpass filtering. For real usage, apply MNE or SciPy filters.
    
    Args:
        eeg_array: shape (num_channels, num_samples)
        low: Low frequency cutoff
        high: High frequency cutoff
        sfreq: Sampling frequency
        
    Returns:
        Filtered EEG array
    """
    return eeg_array

def extract_fft_features(eeg_array):
    """
    Convert EEG to frequency domain using FFT.
    
    Args:
        eeg_array: shape (num_channels, num_samples)
        
    Returns:
        Feature vector combining FFT features from all channels
    """
    feats = []
    for channel in eeg_array:
        fft_vals = np.abs(fft(channel))
        half_spectrum = fft_vals[: len(channel)//2]
        feats.append(half_spectrum)
    return np.hstack(feats)

def extract_psd_features(eeg_array, sfreq=SAMPLING_RATE):
    """
    Extract Power Spectral Density features using Welch's method.
    
    Args:
        eeg_array: shape (num_channels, num_samples)
        sfreq: Sampling frequency
        
    Returns:
        Feature vector combining PSD features from all channels
    """
    feats = []
    for channel in eeg_array:
        freqs, psd = welch(channel, fs=sfreq, nperseg=sfreq//2)
        feats.append(psd)
    return np.hstack(feats)

def extract_wavelet_features(eeg_array, wavelet=WAVELET):
    """
    Extract Discrete Wavelet Transform features.
    
    Args:
        eeg_array: shape (num_channels, num_samples)
        wavelet: Wavelet type to use
        
    Returns:
        Feature vector combining wavelet features from all channels
    """
    feats = []
    for channel in eeg_array:
        coeffs = pywt.wavedec(channel, wavelet=wavelet, level=4)
        wavelet_sub_feats = [np.mean(np.abs(c)) for c in coeffs]
        feats.append(np.array(wavelet_sub_feats))
    return np.hstack(feats)

def get_feature_vector(eeg_array):
    """
    Extract complete feature vector combining multiple transforms.
    
    Args:
        eeg_array: shape (num_channels, num_samples)
        
    Returns:
        Combined feature vector from all transforms
    """
    filtered = bandpass_filter_dummy(eeg_array)
    
    fft_f = extract_fft_features(filtered)
    psd_f = extract_psd_features(filtered)
    wavelet_f = extract_wavelet_features(filtered)
    
    return np.hstack([fft_f, psd_f, wavelet_f])