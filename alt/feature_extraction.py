import numpy as np
import scipy.signal as sps
from scipy.stats import entropy
from antropy import perm_entropy
from scipy.signal import stft

bands = ({'delta' : [1,4],
          'theta' : [4, 8],
          'alpha' : [8, 14],
          'beta' : [14, 30],
          'gamma' : [30,120]})  

# Funktion um Werte pro Band zu bekommen
def get_band_indices(f):
    return {band: np.logical_and(f >= fmin, f <= fmax) for band, (fmin, fmax) in bands.items()}

# Berechne die Power pro Band
def spectral_power(spectrum, idx_dict):
    return [np.nanmean(spectrum[idx]) if spectrum[idx].size > 0 else 0.0 for idx in idx_dict.values()]

# Berechnung der durchschnittlichen spektralen Amplitude
def mean_spectral_amplitude(spectrum, idx_dict):
    spectrum_clipped = np.clip(spectrum, 0, None)
    amp = np.sqrt(spectrum_clipped)
    return [np.nanmean(amp[idx]) if amp[idx].size > 0 else 0.0 for idx in idx_dict.values()]

# Berechnung der spektralen Entropie
def spectral_entropy(f, spectrum, base=np.e):
    psd_norm = spectrum / np.sum(spectrum)
    psd_norm = np.clip(psd_norm, 1e-12, 1)
    log_fn = np.log if base == np.e else np.log2
    entropy_val = -np.sum(psd_norm * log_fn(psd_norm))
    return entropy_val / log_fn(len(psd_norm))

# Berechnung von temporalen Features
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

def feature_extraction_window(signals, fs, stft_window_size, stft_overlap):
    n_channels, n_samples = signals.shape
    nperseg = int(stft_window_size * fs)
    noverlap = int(nperseg * stft_overlap)
    feature_matr = []

    for channel in signals:
        if np.all(channel == 0) or np.isnan(channel).any() or np.isinf(channel).any():
            # Fallback bei defektem Kanal
            feature_matr.append(np.zeros(15))  # 5 Bänder x 2 + 5 extra features
            continue

        f, _, Zxx = stft(channel, fs=fs, nperseg=nperseg, noverlap=noverlap) # Berechnung der spektralen Power
        psd = np.abs(Zxx) ** 2
        avg_spectrum = np.mean(psd, axis=1)
        idx_dict = get_band_indices(f)

        spectral_pwr = spectral_power(avg_spectrum, idx_dict)
        mean_amp = mean_spectral_amplitude(avg_spectrum, idx_dict)

        try:
            hjorth_act, hjorth_mob, hjorth_comp = hjorthparameters(channel)
            spec_ent = spectral_entropy(f, avg_spectrum)
            pfd = petrosian_fd(channel)
        except:
            hjorth_act, hjorth_mob, hjorth_comp, spec_ent, pfd = [0.0] * 5

        features = spectral_pwr + mean_amp + [hjorth_act, hjorth_mob, hjorth_comp, spec_ent, pfd]
        feature_matr.append(np.asarray(features))

    matrix = np.asarray(feature_matr)
    return standardize_matrix(matrix)