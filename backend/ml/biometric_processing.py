import numpy as np
from typing import Dict, List, Tuple
import neurokit2 as nk
from hrvanalysis import get_frequency_domain_features, get_time_domain_features
from scipy.signal import welch

class BiometricProcessor:
    
    def __init__(self, sampling_rate: int = 256):
        self.sampling_rate = sampling_rate
        
    def process_hrv(self, rr_intervals: List[float]) -> Dict[str, float]:
        time_features = get_time_domain_features(rr_intervals)
        freq_features = get_frequency_domain_features(rr_intervals)
        return {**time_features, **freq_features}
    
    def process_blood_pressure(self,
                             systolic: np.ndarray,
                             diastolic: np.ndarray,
                             window_size: int = 10) -> Dict[str, float]:
        features = {
            'systolic_mean': np.mean(systolic),
            'systolic_std': np.std(systolic),
            'diastolic_mean': np.mean(diastolic),
            'diastolic_std': np.std(diastolic),
            'pulse_pressure_mean': np.mean(systolic - diastolic),
            'pulse_pressure_std': np.std(systolic - diastolic)
        }
        
        sys_diff = np.diff(systolic)
        dia_diff = np.diff(diastolic)
        
        features.update({
            'systolic_rate': np.mean(sys_diff),
            'diastolic_rate': np.mean(dia_diff),
            'systolic_rate_std': np.std(sys_diff),
            'diastolic_rate_std': np.std(dia_diff)
        })
        
        return features
    
    def process_gsr(self, gsr_signal: np.ndarray) -> Dict[str, float]:
        cleaned_gsr = nk.eda_clean(gsr_signal, sampling_rate=self.sampling_rate)
        
        features = {
            'gsr_mean': np.mean(cleaned_gsr),
            'gsr_std': np.std(cleaned_gsr),
            'gsr_max': np.max(cleaned_gsr),
            'gsr_min': np.min(cleaned_gsr)
        }
        
        freqs, psd = welch(cleaned_gsr, fs=self.sampling_rate)
        features.update({
            'gsr_low_power': np.mean(psd[freqs < 0.1]),
            'gsr_high_power': np.mean(psd[freqs >= 0.1])
        })
        
        return features
    
    def process_respiratory(self, resp_signal: np.ndarray) -> Dict[str, float]:
        cleaned_resp = nk.rsp_clean(resp_signal, sampling_rate=self.sampling_rate)
        rsp_rate = nk.rsp_rate(cleaned_resp, sampling_rate=self.sampling_rate)
        peaks, _ = nk.rsp_peaks(cleaned_resp, sampling_rate=self.sampling_rate)
        
        features = {
            'resp_rate_mean': np.mean(rsp_rate),
            'resp_rate_std': np.std(rsp_rate),
            'resp_amplitude_mean': np.mean(np.abs(cleaned_resp)),
            'resp_amplitude_std': np.std(np.abs(cleaned_resp)),
            'resp_peaks_per_minute': len(peaks) * (60 / (len(resp_signal) / self.sampling_rate))
        }
        
        return features
    
    def extract_all_features(self,
                           rr_intervals: List[float] = None,
                           blood_pressure: Tuple[np.ndarray, np.ndarray] = None,
                           gsr_signal: np.ndarray = None,
                           resp_signal: np.ndarray = None) -> Dict[str, float]:
        features = {}
        
        if rr_intervals is not None:
            features.update({
                f'hrv_{k}': v for k, v in self.process_hrv(rr_intervals).items()
            })
            
        if blood_pressure is not None:
            systolic, diastolic = blood_pressure
            features.update({
                f'bp_{k}': v for k, v in 
                self.process_blood_pressure(systolic, diastolic).items()
            })
            
        if gsr_signal is not None:
            features.update({
                f'gsr_{k}': v for k, v in self.process_gsr(gsr_signal).items()
            })
            
        if resp_signal is not None:
            features.update({
                f'resp_{k}': v for k, v in 
                self.process_respiratory(resp_signal).items()
            })
            
        return features