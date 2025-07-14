import os
import time
import torch
from wettbewerb import EEGDataset
from new_preprocess import preprocess_signal_with_montages
from window_sequences import window_eeg_data
from feature_extraction import feature_extraction_window

# === Configuration ===
window_step_configs = [(4, 1)]
dataset = EEGDataset("../shared_data/training")
target_fs = 256

# === Loop over configs ===
for window_size, step_size in window_step_configs:
    save_folder = f"data_new_window/win{int(window_size)}_step{int(step_size)}_transformer"
    os.makedirs(save_folder, exist_ok=True)
    print(f"\n🔧 Starting config: window={window_size}s, step={step_size}s")

    total_start = time.time()

    # === Process each EEG ===
    for i in range(len(dataset)):
        start = time.time()

        eeg_id, channels, raw_data, fs, _, label = dataset[i]
        seizure_label, seizure_onset, seizure_offset = label

        try:
            # 1. Preprocessing
            montage_names, processed_signal, montage_missing, new_fs = preprocess_signal_with_montages(
                channels, raw_data, target_fs=target_fs, original_fs=fs, ids=eeg_id
            )
            if montage_missing:
                print(f"⚠️ Skipping {eeg_id} (montage missing)")
                continue

            # 2. Windowing
            windows, labels, timestamps, _ = window_eeg_data(
                processed_signal,
                resampled_fs=new_fs,
                seizure_onset=seizure_onset,
                seizure_offset=seizure_offset,
                window_size=window_size,
                step_size=step_size
            )
            if not windows:
                print(f"⚠️ No valid windows for {eeg_id}")
                continue

            # 3. Feature extraction
            feature_list = []
            for win in windows:
                try:
                    features = feature_extraction_window(
                        win, fs=new_fs, stft_window_size=1.0, stft_overlap=0.5
                    )
                    feature_list.append(torch.tensor(features, dtype=torch.float32))
                except Exception as fe:
                    print(f"⚠️ Feature extraction failed for {eeg_id}: {fe}")
                    continue

            if not feature_list:
                print(f"⚠️ All windows failed for {eeg_id}")
                continue

            # 4. Save as full sequence
            save_data = {
                "eeg_id": eeg_id,
                "windows": torch.stack(feature_list),                          # (T, C, F)
                "window_labels": torch.tensor(labels[:len(feature_list)]),    # (T,)
                "timestamps": timestamps[:len(feature_list)],
                "seizure_onset": seizure_onset,
                "seizure_offset": seizure_offset
            }

            save_path = os.path.join(save_folder, f"{eeg_id}.pt")
            torch.save(save_data, save_path)

            elapsed = time.time() - start
            print(f"[{i+1}/{len(dataset)}] ✅ {eeg_id} — {len(feature_list)} windows — {elapsed:.2f}s", end='\r')

        except Exception as e:
            print(f"❌ Error with {eeg_id}: {e}")

    total_time = time.time() - total_start
    print(f"\n⏱️ Finished in {total_time:.2f} seconds")
    print(f"✅ Data saved to: {save_folder}")