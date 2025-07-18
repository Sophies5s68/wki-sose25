import numpy as np
import scipy.signal as sps
from scipy.stats import entropy
from scipy.signal import stft
import torch

# ---- Datei um die Features zu berechnen zum Erstellen des Trainings Datensatzes ----

# Defintion der verschiedenen Frequenzbänder Delta, Theta, Alpha, Beta und Gamma
bands = ({'delta' : [1,4],
          'theta' : [4, 8],
          'alpha' : [8, 14],
          'beta' : [14, 30],
          'gamma' : [30,120]})  

## Methode um EEG-Signal in überlappende oder nicht-überlappende Fenster aufzuteilen für unsere Datensätze ##
def window_eeg_data(signal, resampled_fs, seizure_onset, seizure_offset, window_size, step_size):
    '''
    Inputs:
    - signal: 2D-Array mit der Form (channels, samples), enthält das EEG-Signal
    - resampled_fs: Abtastrate des Signals in Hz nach Resampling
    - seizure_onset: Zeit in Sekunden, ab wann ein epileptischer Anfall beginnt (0.0 falls keiner vorhanden)
    - seizure_offset: Zeit in Sekunden, ab wann ein Anfall endet (0.0 falls keiner vorhanden)
    - window_size: Fenstergröße in Sekunden
    - step_size: Schrittweise in Sekunden zwischen aufeinanderfolgenden Fenstern.
    
    Outputs:
    - windows: Liste von 2D-Array (channels, window_samples), die Fenster des Signals enthalten
    - labels: Liste von Labels (0= kein Anfall, 1= Anfall), je Fenster entsprechend der Überlappung mit dem Anfall
    - timestamps: Liste von Zeitstempeln für den Start jedes Fensters
    - used_whole_recording: True, wenn das gesamte Signal verwendet wurde, weil es kürzer als Fensterlänge war
    '''
    # Prüfen ob Signal die richtige Form zum Fenstenr hat 
    if signal.ndim != 2:
        raise ValueError("Signal muss 2D sein mit (channels,samples)")
    
    # Umrechnen der Sekunden in Abtastpunkte
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

## Methode um unbekanntes EEG-Signal zu fenstern für die predict.py ##
def window_prediction(signal, resampled_fs, window_size, step_size):
    '''
    Inputs:
    - signal: 2D-Array mit der Form (channels, samples), enthält das EEG-Signal
    - resampled_fs: Abtastrate des Signals in Hz nach Resampling
    - window_size: Fenstergröße in Sekunden
    - step_size: Schrittweite in Sekunden zwischen aufeinanderfolgenden Fenstern

    Outputs:
    - windows: Liste von 2D-Arrays (channels, window_samples), die Fenster des Signals enthalten
    - timestamps: Liste von Zeitstempeln (in Sekunden) für den Start jedes Fensters
    - used_whole_recording: True, wenn das gesamte Signal verwendet wurde, weil es kürzer als die Fensterlänge war
    '''
    if signal.ndim != 2:
        raise ValueError("Signal muss 2D sein mit (channels, samples)")

    window_samples = int(window_size * resampled_fs)
    step_samples = int(step_size * resampled_fs)
    n_channels, n_samples = signal.shape

    windows = []
    timestamps = []
    used_whole_recording = False

    # Falls Signal kürzer als ein Fenster ist: gesamtes Signal verwenden
    if n_samples < window_samples:
        windows.append(signal)
        timestamps.append(0.0)
        used_whole_recording = True
        return windows, timestamps, used_whole_recording

    # Aufteilen des Signals in Fenster
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = signal[:, start:end]
        windows.append(window)
        timestamps.append(start / resampled_fs)

    return windows, timestamps, used_whole_recording


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

# Berechnen der spektralen Entropie
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

## Methode um Features zu berechnen, Spektrum wird mit Welch-Methode berechnet ##
def feature_extraction_window(signals, fs):
    '''
    Extrahiert frequenz- und zeitraumbasierte Merkmale aus einem mehrkanaligen EEG-Fenster.

    Inputs:
    - signals: 2D-Array der Form (n_channels, n_samples), enthält ein EEG-Fenster mit mehreren Kanälen
    - fs: Sampling-Frequenz (Hz) des Signals

    Outputs:
    - matrix: 2D-Array der Form (n_channels, n_features), enthält normalisierte Features pro Kanal

    Beschreibung der Features pro Kanal (insgesamt 15):
    - 5x spektrale Leistungen in definierten Frequenzbändern (z. B. Delta, Theta, Alpha, Beta, Gamma)
    - 5x mittlere spektrale Amplituden in denselben Bändern
    - 3x Hjorth-Parameter:
        - Aktivität
        - Mobilität
        - Komplexität
    - 1x Spektrale Entropie
    - 1x Petrosian Fractal Dimension
    '''
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


## Methode um gruppierte Features des gefensterten EEG-Signals zu extrahieren, wird mit STFT verwendet, um Laufzeit zu verbessern ##
def features_prediction_grouped(signals: np.ndarray, fs: float, stft_window_size: float, stft_overlap: float) -> torch.Tensor:
    '''
    Gibt die gruppierten Features zurück, wie sie für das CNN im Format (4, 5, 6) erwartet werden.
    
    Inputs:
    - signals: (n_channels, n_samples)
    - fs: Sampling Rate
    - stft_window_size: STFT Fenstergröße in Sekunden
    - stft_overlap: Überlappung (0–1)

    Output:
    - torch.Tensor der Form (4 Gruppen, max 5 Features, 6 Kanäle)
    '''

    n_channels, n_samples = signals.shape
    nperseg = int(stft_window_size * fs)
    noverlap = int(nperseg * stft_overlap)
    feature_matrix = []

    for channel in signals:
        if np.all(channel == 0) or np.isnan(channel).any() or np.isinf(channel).any():
            feature_matrix.append(np.zeros(15))  # 5+5+3+1+1 = 15 Features
            continue

        # STFT → Spektrum
        f, _, Zxx = sps.stft(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
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

        # Reihenfolge: 5x Power, 5x Amplitude, 3x Hjorth, 1x Entropie, 1x PFD = 15
        features = spectral_pwr + mean_amp + [hjorth_act, hjorth_mob, hjorth_comp, spec_ent, pfd]
        feature_matrix.append(np.asarray(features))

    feature_matrix = np.asarray(feature_matrix)  # (n_channels, 15)
    feature_matrix = standardize_matrix(feature_matrix)
   
    assert feature_matrix.shape == (6, 15), f"Expected (6,15), got {feature_matrix.shape}"

    # Gruppieren: Transponieren → (15, 6)
    features_t = feature_matrix.T

    # Aufteilen in Gruppen
    groups = [
        features_t[0:5],    # Gruppe 1: Spektral Power
        features_t[5:10],   # Gruppe 2: Mittlere Amplituden
        features_t[10:13],  # Gruppe 3: Hjorth
        features_t[13:15],  # Gruppe 4: Entropie, PFD
    ]

    max_len = max(g.shape[0] for g in groups)  # = 5
    n_channels = groups[0].shape[1]            # = 6

    padded_groups = []
    for g in groups:
        pad_size = max_len - g.shape[0]
        if pad_size > 0:
            pad = np.zeros((pad_size, n_channels), dtype=g.dtype)
            g_padded = np.vstack([g, pad])
        else:
            g_padded = g
        padded_groups.append(torch.tensor(g_padded, dtype=torch.float32))

    # (4, 5, 6)
    return torch.stack(padded_groups, dim=0)


'''
## Methode um Features des gefensterten EEG-Signals zu extrahieren, wird mit STFT verwendet, um Laufzeit zu verbessern ##
# nicht mehr in Verwendung
def features_prediction(signals, fs, stft_window_size, stft_overlap):
    
    Extrahiert frequenz- und zeitraumbasierte Merkmale aus einem mehrkanaligen EEG-Fenster.
    
    Inputs:
    - signals: 2D-Array der Form (n_channels, n_samples), enthält das EEG-Fenster
    - fs: Sampling-Frequenz (Hz) des Signals
    - stft_window_size: Fenstergröße für die STFT (in Sekunden)
    - stft_overlap: Überlappung der STFT-Fenster (zwischen 0 und 1)

    Outputs:
    - matrix: 2D-Array der Form (n_channels, n_features), enthält extrahierte und normalisierte Features pro Kanal

    Beschreibung der Features pro Kanal:
    - 5x spektrale Leistungen in typischen EEG-Bändern (z.B. Delta, Theta, Alpha, Beta, Gamma)
    - 5x mittlere spektrale Amplituden für dieselben Bänder
    - 3x Hjorth-Parameter (Aktivität, Mobilität, Komplexität)
    - 1x spektrale Entropie
    - 1x Petrosian Fractal Dimension
    
    
    n_channels, n_samples = signals.shape
    nperseg = int(stft_window_size * fs)
    noverlap = int(nperseg * stft_overlap)
    feature_matr = []

    for channel in signals:
        if np.all(channel == 0) or np.isnan(channel).any() or np.isinf(channel).any(): # Falls Kanal fehlerhaft ist, wird er mit einem Null-Vektor ersetzt
            feature_matr.append(np.zeros(15))  # 5 Bänder x 2 + 5 extra features
            continue
        # Berechnung der STFT
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
'''