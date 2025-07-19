import os
import time
import torch
from wettbewerb import EEGDataset
from new_preprocess import preprocess_signal_with_montages
from new_features import window_eeg_data
from faster_features import feature_extraction_window

## ---- Datei um Daten für ein Transformer Modell zu laden ----

# === Konfiguration ===
save_root = "data_new_window"
window_size = 4.0         # Fenstergröße in Sekunden
step_size = 1.0           # Schrittweite in Sekunden
target_fs = 256           # Zielabtastrate in Hz
stft_window_size = 1.0    # Fenstergröße für STFT
stft_overlap = 0.5        # Überlappung für STFT

# EEG-Datensatz laden
dataset = EEGDataset("../shared_data/training")

# Zielverzeichnis zum Speichern erstellen
save_folder = f"{save_root}/win{int(window_size)}_step{int(step_size)}_transformer"
os.makedirs(save_folder, exist_ok=True)

print(f"\nSpeichern in: {save_folder}")
total_start = time.time()

# Schleife über alle EEG-Daten
for i in range(len(dataset)):
    eeg_start = time.time()

    # === EEG-Daten laden ===
    eeg_id, channels, raw_data, fs, _, label = dataset[i]
    seizure_label, seizure_onset, seizure_offset = label

    # === Vorverarbeitung ===
    try:
        montage_names, processed_signal, montage_missing, new_fs = preprocess_signal_with_montages(
            channels, raw_data, target_fs=target_fs, original_fs=fs, ids=eeg_id
        )
    except Exception as e:
        print(f"Vorverarbeitung fehlgeschlagen für {eeg_id}: {e}")
        continue

    if montage_missing:
        print(f"Überspringe {eeg_id} (Montage fehlt)")
        continue

    # === Fensterung des Signals ===
    windows, labels, timestamps, _ = window_eeg_data(
        processed_signal,
        resampled_fs=new_fs,
        seizure_onset=seizure_onset,
        seizure_offset=seizure_offset,
        window_size=window_size,
        step_size=step_size
    )

    if not windows:
        print(f"Keine gültigen Fenster für {eeg_id}")
        continue

    # === Merkmalsextraktion für alle Fenster ===
    window_tensors = []
    for win in windows:
        try:
            features = feature_extraction_window(
                win, fs=new_fs,
                stft_window_size=stft_window_size,
                stft_overlap=stft_overlap
            )  # Ausgabeform: (Kanäle, Merkmale)
            window_tensors.append(torch.tensor(features, dtype=torch.float32))
        except Exception as e:
            print(f"Merkmalsextraktion fehlgeschlagen für {eeg_id}, Fenster wird übersprungen: {e}")
            continue

    if not window_tensors:
        print(f"Alle Merkmalsfenster ungültig für {eeg_id}")
        continue

    # === Zusammenstellen und Speichern ===
    try:
        save_data = {
            "eeg_id": eeg_id,
            "windows": torch.stack(window_tensors),                            # (T, C, F)
            "window_labels": torch.tensor(labels, dtype=torch.long),          # (T,)
            "timestamps": timestamps,                                         # Liste von Zeitpunkten
            "seizure_onset": seizure_onset,
            "seizure_offset": seizure_offset
        }

        save_path = os.path.join(save_folder, f"{eeg_id}.pt")
        torch.save(save_data, save_path)

        eeg_time = time.time() - eeg_start
        print(f"[{i+1}/{len(dataset)}] {eeg_id} — {len(labels)} Fenster — {eeg_time:.2f}s", end='\r')

    except Exception as e:
        print(f"Fehler beim Speichern von {eeg_id}: {e}")

total_time = time.time() - total_start
print(f"\n\nAlle Sequenzen wurden gespeichert unter: {save_folder}")
print(f"Gesamtdauer: {total_time:.2f} Sekunden")
