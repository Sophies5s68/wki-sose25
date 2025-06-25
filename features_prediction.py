import numpy as np
import scipy.signal as sps
from joblib import Parallel, delayed
from scipy.stats import entropy
from antropy import perm_entropy

# Frequenzbänder
bands = {'delta': [1, 4], 'theta': [4, 8], 'alpha': [8, 14],
         'beta': [14, 30], 'gamma': [30, 120]}

def window_prediction(signal, resampled_fs, window_size, step_size):

    # Testet, ob Input in der richtigen Form ist 
    if signal.ndim !=2: 
        raise ValueError("Signal muss 2D sein, der Größe (n_channels, n_samples)")
    window_samples = int(window_size * resampled_fs)
    step_samples = int(step_size * resampled_fs)
    n_channels, n_samples = signal.shape
    
    windows = []
    timestamps = []
    
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = signal[:, start:end]
        windows.append(window)
        start_sec = start / resampled_fs
        end_sec = end / resampled_fs
        timestamps.append(start_sec)

    return windows, timestamps

def get_band_indices(f):
    return {band: np.logical_and(f >= fmin, f <= fmax) for band, (fmin, fmax) in bands.items()}

def spectral_power_v(spectrum, idx_dict):
    return np.stack([np.nanmean(spectrum[:, idx], axis=1) for idx in idx_dict.values()], axis=1)

def mean_spectral_amplitude_v(spectrum, idx_dict):
    amp = np.sqrt(np.clip(spectrum, 0, None))
    return np.stack([np.nanmean(amp[:, idx], axis=1) for idx in idx_dict.values()], axis=1)

def hjorth_parameters_v(sig):
    diff1 = np.diff(sig, axis=1)
    diff2 = np.diff(diff1, axis=1)
    activity = np.var(sig, axis=1)
    mobility = np.sqrt(np.var(diff1, axis=1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2, axis=1) / (np.var(diff1, axis=1) + 1e-10)) / (mobility + 1e-10)
    return np.stack([activity, mobility, complexity], axis=1)

def spectral_entropy_channel(f, spectrum_ch, base=np.e):
    psd_norm = spectrum_ch / np.sum(spectrum_ch)
    psd_norm = np.clip(psd_norm, 1e-12, 1)
    log_fn = np.log if base == np.e else np.log2
    return -np.sum(psd_norm * log_fn(psd_norm)) / log_fn(len(psd_norm))

def petrosian_fd(sig):
    N = len(sig)
    diff = np.diff(sig)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta))) if N_delta != 0 else 0.0

def additional_features_parallel(signals, f, spectrum):
    def compute(ch_idx):
        spec_ent = spectral_entropy_channel(f, spectrum[ch_idx])
        pfd = petrosian_fd(signals[ch_idx])
        return [spec_ent, pfd]
    return np.array(Parallel(n_jobs=-1)(delayed(compute)(i) for i in range(signals.shape[0])))

def standardize_matrix(matr):
    mean = matr.mean(axis=0)
    std = matr.std(axis=0)
    std[std == 0] = 1
    return (matr - mean) / std

def feature_extraction_window(signals, fs):
    # Welch auf allen Kanälen
    f, spectrum = sps.welch(signals, fs, axis=-1)
    idx_dict = get_band_indices(f)

    # Features vektorisieren
    spectral_pwr = spectral_power_v(spectrum, idx_dict)
    mean_amp = mean_spectral_amplitude_v(spectrum, idx_dict)
    hjorth = hjorth_parameters_v(signals)

    # Nicht vektorisierbare Features parallelisieren
    extra_feats = additional_features_parallel(signals, f, spectrum)

    # Alles zusammenführen
    features = np.concatenate([spectral_pwr, mean_amp, hjorth, extra_feats], axis=1)
    return standardize_matrix(features)
