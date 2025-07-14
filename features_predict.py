import numpy as np
import scipy.signal as sps
from scipy.stats import entropy
from scipy.signal import stft

# Datei zum extrahieren der Features während prediction.py

bands = ({'delta' : [1,4],
          'theta' : [4, 8],
          'alpha' : [8, 14],
          'beta' : [14, 30],
          'gamma' : [30,120]})  


# Methode um EEG Signal in Windows einzuteilen
def window_eeg_data(signal, resampled_fs, seizure_onset, seizure_offset, window_size, step_size):

    # Testet, ob Input in der richtigen Form ist 
    if signal.ndim !=2: 
        raise ValueError("Signal muss 2D sein, der Größe (n_channels, n_samples)")
    window_samples = int(window_size * resampled_fs)
    step_samples = int(step_size * resampled_fs)
    n_channels, n_samples = signal.shape
    
    windows = []
    labels = []
    timestamps = []
    
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = signal[:, start:end]
        windows.append(window)
        start_sec = start / resampled_fs
        end_sec = end / resampled_fs
        timestamps.append(start_sec)

        # Window bekommt Lable 1, wenn Anfall vorhanden ist, sonst 0
        if end_sec >= seizure_onset and start_sec <= seizure_offset:
            labels.append(1)
        else:
            labels.append(0)

    return windows, labels, timestamps

# Berechnen der Indizes
def get_band_indices(f):
    return {band: np.logical_and(f >= fmin, f <= fmax) for band, (fmin, fmax) in bands.items()}

# Berechne die Power pro Band
def spectral_power(spectrum, idx_dict):
    return [np.nanmean(spectrum[idx]) if spectrum[idx].size > 0 else 0.0 for idx in idx_dict.values()]

# Berechnen der durchschnittlichen Amplitude
def mean_spectral_amplitude(spectrum, idx_dict):
    spectrum_clipped = np.clip(spectrum, 0, None)
    amp = np.sqrt(spectrum_clipped)
    return [np.nanmean(amp[idx]) if amp[idx].size > 0 else 0.0 for idx in idx_dict.values()]

# Berechnen der spekatralen Entropie
def spectral_entropy(f, spectrum, base=np.e):
    psd_norm = spectrum / np.sum(spectrum)
    psd_norm = np.clip(psd_norm, 1e-12, 1)
    log_fn = np.log if base == np.e else np.log2
    entropy_val = -np.sum(psd_norm * log_fn(psd_norm))
    return entropy_val / log_fn(len(psd_norm))

# Berechnen der Hjorth Parameter
def hjorthparameters(sig):
    sig = np.asarray(sig)
    first_deriv = np.diff(sig)
    second_deriv = np.diff(first_deriv)
    activity = np.var(sig)
    mobility = np.sqrt(np.var(first_deriv) / (activity + 1e-10))
    complexity = np.sqrt(np.var(second_deriv) / (np.var(first_deriv) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity

# Berechnen der funktionalen Dispersion
def petrosian_fd(sig):
    N = len(sig)
    diff = np.diff(sig)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta))) if N_delta != 0 else 0.0

# Über Features standardisieren um bessere Eingaben für CNN zu erzeugen
def standardize_matrix(matr):
    mean = matr.mean(axis=0)
    std = matr.std(axis=0)
    std[std == 0] = 1
    return (matr - mean) / std

# Methode um Features zu berechnen
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

        f, _, Zxx = stft(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
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

# Methode um zu klassifizierendes Signal zu fenstern
def window_prediction(signal, resampled_fs, window_size, step_size):
    # Testet, ob Input in der richtigen Form ist 
    if signal.ndim != 2: 
        raise ValueError("Signal muss 2D sein, der Größe (n_channels, n_samples)")

    window_samples = int(window_size * resampled_fs)
    step_samples = int(step_size * resampled_fs)
    n_channels, n_samples = signal.shape

    windows = []
    timestamps = []

    if n_samples < window_samples:
        # Signal auf Fenstergröße mit Nullen am Ende auffüllen
        pad_width = window_samples - n_samples
        padded_signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
        windows.append(padded_signal)
        timestamps.append(0.0)
        return windows, timestamps

    # Sliding-Window-Verfahren
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = signal[:, start:end]
        windows.append(window)
        timestamps.append(start / resampled_fs)
    
    if not windows:
        padded_signal = np.pad(signal, ((0, 0), (0, window_samples - n_samples)), mode='constant')
        windows = [padded_signal]
        timestamps = [0.0]
        
    return windows, timestamps

'''
# Methode um Spektrogram zu berechnen
# wird nicht mehr verwendet
def compute_spectrogram(window, n_fft=128, hop_length=64, win_length=128):
    channels, samples = window.shape
    specs = []
    window_fn = torch.hann_window(win_length)
    
    for ch in range(channels):
        stft_res = torch.stft(window[ch], n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                              window=window_fn, return_complex=True)
        mag = torch.abs(stft_res)
        specs.append(torch.log1p(mag))
        
    specs = torch.stack(specs)  # [channels, freq_bins, time_bins]
    return specs

'''