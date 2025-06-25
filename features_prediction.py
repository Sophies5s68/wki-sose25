import numpy as np
import scipy.signal as sps
from joblib import Parallel, delayed
from scipy.stats import entropy
from antropy import perm_entropy

# Frequenzbänder
bands = {'delta': [1, 4], 'theta': [4, 8], 'alpha': [8, 14],
         'beta': [14, 30], 'gamma': [30, 120]}

def get_band_indices(f):
    return {band: np.logical_and(f >= fmin, f <= fmax) for band, (fmin, fmax) in bands.items()}

# Berechne die Power pro Band
def spectral_power(spectrum, idx_dict):
    return [np.nanmean(spectrum[idx]) if spectrum[idx].size > 0 else 0.0 for idx in idx_dict.values()]

def mean_spectral_amplitude(spectrum, idx_dict):
    spectrum_clipped = np.clip(spectrum, 0, None)
    amp = np.sqrt(spectrum_clipped)
    return [np.nanmean(amp[idx]) if amp[idx].size > 0 else 0.0 for idx in idx_dict.values()]

def spectral_entropy(f, spectrum, base=np.e):
    psd_norm = spectrum / np.sum(spectrum)
    psd_norm = np.clip(psd_norm, 1e-12, 1)
    log_fn = np.log if base == np.e else np.log2
    entropy_val = -np.sum(psd_norm * log_fn(psd_norm))
    return entropy_val / log_fn(len(psd_norm))

def hjorthparameters(sig):
    sig = np.asarray(sig)
    first_deriv = np.diff(sig)
    second_deriv = np.diff(first_deriv)
    activity = np.var(sig)
    mobility = np.sqrt(np.var(first_deriv) / (activity + 1e-10))
    complexity = np.sqrt(np.var(second_deriv) / (np.var(first_deriv) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity

def petrosian_fd(sig):
    N = len(sig)
    diff = np.diff(sig)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta))) if N_delta != 0 else 0.0

def standardize_matrix(matr):
    mean = matr.mean(axis=0)
    std = matr.std(axis=0)
    std[std == 0] = 1
    return (matr - mean) / std


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

def feature_extraction_window(signals, fs):
    signals = signals.astype(np.float32)
    n_channels, n_samples = signals.shape
    nperseg = n_samples  # Volle Länge des Windows

    # STFT berechnen
    f, t, Zxx = sps.stft(signals, fs=fs, nperseg=nperseg, noverlap=0, axis=-1, boundary=None)
    spectrum = np.abs(Zxx) ** 2  # Power Spectrum (n_channels, freq_bins, time_frames)
    spectrum_mean = np.mean(spectrum, axis=-1)  # Mittelung über Zeitachse → (n_channels, freq_bins)

    idx_dict = get_band_indices(f)

    # Features pro Kanal berechnen
    feature_list = []
    for ch in range(n_channels):
        band_power = spectral_power(spectrum_mean[ch], idx_dict)
        mean_amp = mean_spectral_amplitude(spectrum_mean[ch], idx_dict)

        activity, mobility, complexity = hjorthparameters(signals[ch])
        hjorth = [activity, mobility, complexity]

        spec_ent = spectral_entropy(f, spectrum_mean[ch])
        pfd = petrosian_fd(signals[ch])

        all_feats = band_power + mean_amp + hjorth + [spec_ent, pfd]
        feature_list.append(all_feats)


    features = np.array(feature_list)  # Shape: (n_channels, n_features)
    return standardize_matrix(features)


