import os
import time
import torch
from wettbewerb import EEGDataset
from new_preprocess import preprocess_signal_with_montages
from window_sequences import window_eeg_data
from feature_extraction import feature_extraction_window

## ---- Datei um Daten für ein Transformer Modell zu laden ----

# === Konfiguration ===
window_step_configs = [(4, 1)]  # Liste von Konfigurationen: Fenstergröße 4 Sekunden, Schrittweite 1 Sekunde
dataset = EEGDataset("../shared_data/training")  # EEG-Datensatz laden
target_fs = 256  # Zielabtastrate (Hz)

# === Schleife über die Konfigurationen ===
for window_size, step_size in window_step_configs:
    save_folder = f"data_new_window/win{int(window_size)}_step{int(step_size)}_transformer"
    os.makedirs(save_folder, exist_ok=True)
    print(f"\n🔧 Starte Konfiguration: Fenster={window_size}s, Schritt={step_size}s")

    total_start = time.time()  # Gesamtzeitmessung starten

    # === Schleife über alle EEGs im Datensatz ===
    for i in range(len(dataset)):
        start = time.time()  # Zeitmessung für dieses EEG

        # EEG-Daten abrufen
        eeg_id, channels, raw_data, fs, _, label = dataset[i]
        seizure_label, seizure_onset, seizure_offset = label

        try:
            # 1. Vorverarbeitung (z.B. Filterung, Montage anpassen)
            montage_names, processed_signal, montage_missing, new_fs = preprocess_signal_with_montages(
                channels, raw_data, target_fs=target_fs, original_fs=fs, ids=eeg_id
            )
            if montage_missing:
                print(f" Überspringe {eeg_id} (Montage fehlt)")
                continue  # Überspringt EEG, wenn keine gültige Montage erstellt werden konnte

            # 2. Fensterung der Daten
            windows, labels, timestamps, _ = window_eeg_data(
                processed_signal,
                resampled_fs=new_fs,
                seizure_onset=seizure_onset,
                seizure_offset=seizure_offset,
                window_size=window_size,
                step_size=step_size
            )
            if not windows:
                print(f" Keine gültigen Fenster für {eeg_id}")
                continue  # Falls keine Fenster erzeugt wurden, überspringen

            # 3. Merkmalsextraktion für jedes Fenster
            feature_list = []
            for win in windows:
                try:
                    features = feature_extraction_window(
                        win, fs=new_fs, stft_window_size=1.0, stft_overlap=0.5
                    )
                    feature_list.append(torch.tensor(features, dtype=torch.float32))
                except Exception as fe:
                    print(f" Merkmalsextraktion fehlgeschlagen für {eeg_id}: {fe}")
                    continue  # Einzelne fehlerhafte Fenster überspringen

            if not feature_list:
                print(f" Alle Fenster fehlgeschlagen für {eeg_id}")
                continue  # Wenn keine Features extrahiert wurden, überspringen

            # 4. Speichern der sequenziellen Fensterdaten
            save_data = {
                "eeg_id": eeg_id,
                "windows": torch.stack(feature_list),                          # (T, C, F) — T Fenster, C Kanäle, F Merkmale
                "window_labels": torch.tensor(labels[:len(feature_list)]),    # (T,) — Labels pro Fenster
                "timestamps": timestamps[:len(feature_list)],                 # Zeitstempel der Fenster
                "seizure_onset": seizure_onset,
                "seizure_offset": seizure_offset
            }

            save_path = os.path.join(save_folder, f"{eeg_id}.pt")
            torch.save(save_data, save_path)  # Speichern als .pt-Datei

            elapsed = time.time() - start
            print(f"[{i+1}/{len(dataset)}]  {eeg_id} — {len(feature_list)} Fenster — {elapsed:.2f}s", end='\r')

        except Exception as e:
            print(f" Fehler bei {eeg_id}: {e}")

    total_time = time.time() - total_start
    print(f"\n Fertig in {total_time:.2f} Sekunden")
    print(f" Daten gespeichert unter: {save_folder}")
