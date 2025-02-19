"""EEG processing utilities for feature extraction and embedding generation."""

import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch, butter, filtfilt
import pywt

SAMPLING_RATE = 256
WAVELET = 'db4'

def bandpass_filter(eeg_array, low=0.5, high=40, sfreq=256):
    nyq = sfreq * 0.5
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, eeg_array, axis=1)

def extract_features(eeg_array, sfreq=SAMPLING_RATE, wavelet=WAVELET):
    """Extract combined feature vector using FFT, PSD, and wavelets."""
    filtered = bandpass_filter(eeg_array)
    features = []
    
    for channel in filtered:
        # FFT features
        fft_vals = np.abs(fft(channel))
        features.append(fft_vals[: len(channel)//2])
        
        # PSD features
        _, psd = welch(channel, fs=sfreq, nperseg=sfreq//2)
        features.append(psd)
        
        # Wavelet features
        coeffs = pywt.wavedec(channel, wavelet=wavelet, level=4)
        wavelet_feats = [np.mean(np.abs(c)) for c in coeffs]
        features.append(np.array(wavelet_feats))
    
    return np.hstack(features)

# Alias for backward compatibility
get_feature_vector = extract_features