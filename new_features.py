import numpy as np
import scipy.signal as sps
from scipy.stats import entropy


# Datei um die Features zu berechnen zum Erstellen des Trainings Datensatzes

bands = ({'delta' : [1,4],
          'theta' : [4, 8],
          'alpha' : [8, 14],
          'beta' : [14, 30],
          'gamma' : [30,120]})  

# Methode um EEG Signal in Windows einzuteilen
def window_eeg_data(signal, resampled_fs, seizure_onset, seizure_offset, window_size, step_size):

    if signal.ndim != 2:
        raise ValueError("Signal muss 2D sein mit (channels,samples)")

    window_samples = int(window_size * resampled_fs)
    step_samples = int(step_size * resampled_fs)
    n_channels, n_samples = signal.shape

    windows = []
    labels = []
    timestamps = []
    used_whole_recording = False

    # Prüfen ob Anfall vorhanden
    has_valid_seizure = not (seizure_onset == 0.0 and seizure_offset == 0.0)

    # Falls Signal kürzer als ein Fenster ist das gesamte Signal nutzen
    if n_samples < window_samples:
        windows.append(signal)
        timestamps.append(0.0)
        used_whole_recording = True

        start_sec = 0.0
        end_sec = n_samples / resampled_fs

        if has_valid_seizure:
            label = 1 if (end_sec >= seizure_onset and start_sec <= seizure_offset) else 0
        else:
            label = 0
        labels.append(label)
        return windows, labels, timestamps, used_whole_recording

    # Aufteilen des Signals in Fenster
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = signal[:, start:end]
        windows.append(window)

        start_sec = start / resampled_fs
        end_sec = end / resampled_fs
        timestamps.append(start_sec)

        if has_valid_seizure:
            label = 1 if (end_sec >= seizure_onset and start_sec <= seizure_offset) else 0
        else:
            label = 0

        labels.append(label)

    return windows, labels, timestamps, used_whole_recording

'''
# Windowing mit kleineren Windows während Anfällen um Samples für Anfälle zu steigern
# nicht mehr verwendet

def window_eeg_data_variable(signal, resampled_fs, seizure_onset, seizure_offset, 
                              normal_window_size, normal_step_size,
                              seizure_window_size, seizure_step_size):
    """
    Fenster EEG-Daten mit variabler Fenstergröße: kleinere Fenster während Anfällen.

    Returns:
    - windows: Liste mit nparrays, mit der shape (n_channels, window_samples)
    - labels: Liste mit 0 (kein Anfall) und 1 (Anfall)
    - timestamps: Liste mit Zeiten von den Fenstern in Sekunden
    """
    if signal.ndim != 2: 
        raise ValueError("Signal muss 2D sein, der Größe (n_channels, n_samples)")

    n_channels, n_samples = signal.shape
    windows, labels, timestamps = [], [], []

    current_time = 0.0
    while True:
        # Entscheide, ob wir im Anfall sind oder nicht
        if seizure_onset <= current_time <= seizure_offset:
            window_size = seizure_window_size
            step_size = seizure_step_size
        else:
            window_size = normal_window_size
            step_size = normal_step_size

        window_samples = int(window_size * resampled_fs)
        step_samples = int(step_size * resampled_fs)
        start = int(current_time * resampled_fs)
        end = start + window_samples
        
        if end > n_samples:
            break

        window = signal[:, start:end]
        windows.append(window)
        timestamps.append(current_time)

        # Setze Label
        if (current_time + window_size) >= seizure_onset and current_time <= seizure_offset:
            labels.append(1)
        else:
            labels.append(0)

        current_time += step_size

    return windows, labels, timestamps
'''
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
def feature_extraction_window(signals, fs):
    feature_matr = []
    for channel in signals:
        f, spectrum = sps.welch(channel, fs)
        idx_dict = get_band_indices(f)

        spectral_pwr = spectral_power(spectrum, idx_dict)
        mean_amp = mean_spectral_amplitude(spectrum, idx_dict)
        hjorth_act, hjorth_mob, hjorth_comp = hjorthparameters(channel)
        spec_ent = spectral_entropy(f, spectrum)
        pfd = petrosian_fd(channel)

        features = spectral_pwr + mean_amp + [hjorth_act, hjorth_mob, hjorth_comp, spec_ent, pfd]
        feature_matr.append(np.asarray(features))

    return standardize_matrix(np.asarray(feature_matr))

