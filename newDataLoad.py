from wettbewerb import EEGDataset
from new_preprocess import preprocess_signal_with_montages
from new_features import window_eeg_data, feature_extraction_window
import os, torch
import numpy as np

# Datei zum erstellen von Datensätzen, die anschließend für das Training verwednet werden

# Definieren der Fensterparameter (window_size,step_size)
window_step_configs = [(4,1)]

# Lade das Datenset
dataset = EEGDataset("../shared_data/training")

for window_size, step_size in window_step_configs:
    # Zielordner für extrahierte Daten
    save_folder = f"add_dataset/win{window_size}_step{step_size}"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\nStarting config: window={window_size}s, step={step_size}s")
    # Iteration über alle EEG-Dateien im Datensatz
    for i in range(len(dataset)):
        # Hole Rohdaten aus dem Datensatz
        eeg_id, channels, raw_data, fs, _, label = dataset[i]
        seizure_label, seizure_onset, seizure_offset = label

        # 1. Preprocess: Montage anwenden + Resampling auf Ziel-Frequenz (256 Hz)
        montage_names, processed_signal, montage_missing, new_fs = preprocess_signal_with_montages(
            channels, raw_data, target_fs=256, original_fs=fs, ids=eeg_id
        )
        # Wenn Montage nicht angewendet werden konnte, überspringen
        if montage_missing:
            print(f"Skipping {eeg_id} (montage missing)")
            continue

        # 2. Signal wird in überlappende Zeitfenster geteilt
        windows, labels, timestamps, idk = window_eeg_data(
            processed_signal, resampled_fs=new_fs,
            seizure_onset=seizure_onset,
            seizure_offset=seizure_offset,
            window_size=window_size,
            step_size=step_size
        )

        # 3. Feature extraction per window
        for idx, (window, lbl, ts) in enumerate(zip(windows, labels, timestamps)):
            features = feature_extraction_window(window, new_fs)
            # Speicherpfad bauen und als .pt-Datei sichern
            save_path = os.path.join(save_folder, f"{eeg_id}_win{idx}_lbl{lbl}.pt")
            torch.save((features, lbl, eeg_id, ts, seizure_onset), save_path)

        print(f"[{i+1}/{len(dataset)}] Processed {eeg_id} with {len(windows)} windows.", end='\r')

    print(f"Finished: {save_folder}")