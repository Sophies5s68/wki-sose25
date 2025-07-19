import os
import torch
import numpy as np
from wettbewerb import EEGDataset
from new_preprocess import preprocess_signal_with_montages
from new_features import window_eeg_data
from faster_features import compute_spectrogram
import time

# === Konfiguration ===
window_size = 4   # Fenstergröße in Sekunden
step_size = 2     # Schrittweite in Sekunden
target_fs = 256   # Zielabtastrate in Hz
save_folder = f"raw_dataset/sequences_spectrograms/win{window_size}_step{step_size}"
os.makedirs(save_folder, exist_ok=True)

# Parameter für das Spektrogramm
n_fft = 64
hop_length = 32

def compute_spectrogram(window, sample_rate=target_fs, n_fft=n_fft, hop_length=hop_length):
    """
    Berechnet ein Spektrogramm (STFT) für jedes EEG-Kanalfenster.
    Eingabe: window [Kanäle, Samples]
    Ausgabe: Spektrogramm [Kanäle, Frequenzbins, Zeitbins]
    """
    window_tensor = torch.tensor(window, dtype=torch.float32)  # [C, S]
    specs = []

    for ch in window_tensor:
        stft = torch.stft(
            ch,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft),
            return_complex=True  # komplexe STFT
        )
        magnitude = stft.abs()                  # Betrag der komplexen STFT
        log_spec = torch.log1p(magnitude)       # Logarithmische Skalierung
        specs.append(log_spec)                  # [Frequenz, Zeit]

    return torch.stack(specs)  # [Kanäle, Frequenzbins, Zeitbins]

# Lade EEG-Datensatz
dataset = EEGDataset("../shared_data/training")
start_time = time.time()

for i in range(len(dataset)):
    # Lade einzelne EEG-Datei
    eeg_id, channels, raw_data, fs, _, label = dataset[i]
    seizure_label, seizure_onset, seizure_offset = label

    # Vorverarbeitung (z. B. Montage anwenden, Resampling)
    montage_names, processed_signal, montage_missing, new_fs = preprocess_signal_with_montages(
        channels, raw_data, target_fs=target_fs, original_fs=fs, ids=eeg_id
    )
    if montage_missing:
        print(f"Skipping {eeg_id} (montage missing)")
        continue

    # Fensterung der Zeitreihe
    windows, labels, timestamps, used_whole_recording = window_eeg_data(
        processed_signal,
        resampled_fs=new_fs,
        seizure_onset=seizure_onset,
        seizure_offset=seizure_offset,
        window_size=window_size,
        step_size=step_size
    )
    if used_whole_recording:
        print(f" Used whole recording as one window for patient {eeg_id}")

    # Berechne Spektrogramme für alle Fenster
    spectrogram_windows = []
    for window in windows:
        spec = compute_spectrogram(window)
        # Normalisierung jedes Spektrogramms (pro Kanal)
        spec_norm = (spec - spec.mean(dim=(1, 2), keepdim=True)) / (spec.std(dim=(1, 2), keepdim=True) + 1e-8)
        spectrogram_windows.append(spec_norm)
    spectrogram_tensor = torch.stack(spectrogram_windows)

    # Label für das ganze EEG (nicht pro Fenster)
    sequence_label = seizure_label

    # Speichern als .pt-Datei
    save_path = os.path.join(save_folder, f"{eeg_id}_seq.pt")
    torch.save({
        "windows": spectrogram_tensor,        # Tensorliste [Kanäle, Frequenzbins, Zeitbins]
        "label": sequence_label,
        "window_labels": torch.tensor(labels, dtype=torch.long),
        "eeg_id": eeg_id,
        "timestamps": timestamps,
        "seizure_onset": seizure_onset,
        "seizure_offset": seizure_offset
    }, save_path)

    print(f"[{i+1}/{len(dataset)}] Saved {eeg_id} with {len(windows)} spectrogram windows.", end='\r')

print(f"\nDone. Time elapsed: {time.time() - start_time:.2f} seconds")
