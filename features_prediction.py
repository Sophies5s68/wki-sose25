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
    total_power = np.sum(spectrum)
    if total_power < 1e-12:
        return 0.0
    psd_norm = spectrum / total_power
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
    matr = np.nan_to_num(matr, nan=0.0, posinf=0.0, neginf=0.0)
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

from joblib import Parallel, delayed

def feature_extraction_window(signals, fs):
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    signals = signals.astype(np.float32)
    f, spectrum = compute_power_spectrum(signals, fs)
    idx_dict = get_band_indices(f)

    def process_channel(ch):
        band_power = spectral_power(spectrum[ch], idx_dict)
        mean_amp = mean_spectral_amplitude(spectrum[ch], idx_dict)
        activity, mobility, complexity = hjorthparameters(signals[ch])
        spec_ent = spectral_entropy(f, spectrum[ch])
        pfd = petrosian_fd(signals[ch])
        return band_power + mean_amp + [activity, mobility, complexity, spec_ent, pfd]

    features = Parallel(n_jobs=-1)(delayed(process_channel)(ch) for ch in range(signals.shape[0]))
    return standardize_matrix(np.array(features))


def compute_power_spectrum(signal, fs):
    # signal: (n_channels, n_samples)
    fft_vals = np.fft.rfft(signal, axis=-1)
    spectrum = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(signal.shape[1], d=1/fs)
    return freqs, spectrum

