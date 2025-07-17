import os
import time
import torch
from wettbewerb import EEGDataset
from new_preprocess import preprocess_signal_with_montages
from new_features import window_eeg_data
from faster_features import feature_extraction_window

# === Configuration ===
save_root = "data_new_window"
window_size = 4.0         # seconds
step_size = 1.0           # seconds
target_fs = 256           # Hz
stft_window_size = 1.0    # seconds
stft_overlap = 0.5

# Initialize dataset
dataset = EEGDataset("../shared_data/training")

# Create save directory
save_folder = f"{save_root}/win{int(window_size)}_step{int(step_size)}_transformer"
os.makedirs(save_folder, exist_ok=True)

print(f"\n📁 Saving to: {save_folder}")
total_start = time.time()

for i in range(len(dataset)):
    eeg_start = time.time()

    # === Load EEG ===
    eeg_id, channels, raw_data, fs, _, label = dataset[i]
    seizure_label, seizure_onset, seizure_offset = label

    # === Preprocessing ===
    try:
        montage_names, processed_signal, montage_missing, new_fs = preprocess_signal_with_montages(
            channels, raw_data, target_fs=target_fs, original_fs=fs, ids=eeg_id
        )
    except Exception as e:
        print(f"❌ Failed preprocessing for {eeg_id}: {e}")
        continue

    if montage_missing:
        print(f"⚠️ Skipping {eeg_id} (montage missing)")
        continue

    # === Windowing ===
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

    # === Feature Extraction for All Windows ===
    window_tensors = []
    for win in windows:
        try:
            features = feature_extraction_window(
                win, fs=new_fs,
                stft_window_size=stft_window_size,
                stft_overlap=stft_overlap
            )  # (C, F)
            window_tensors.append(torch.tensor(features, dtype=torch.float32))
        except Exception as e:
            print(f"⚠️ Feature extraction failed for {eeg_id}, window skipped: {e}")
            continue

    if not window_tensors:
        print(f"⚠️ All feature windows invalid for {eeg_id}")
        continue

    # === Assemble and Save ===
    try:
        save_data = {
            "eeg_id": eeg_id,
            "windows": torch.stack(window_tensors),                            # (T, C, F)
            "window_labels": torch.tensor(labels, dtype=torch.long),          # (T,)
            "timestamps": timestamps,                                         # List of float
            "seizure_onset": seizure_onset,
            "seizure_offset": seizure_offset
        }

        save_path = os.path.join(save_folder, f"{eeg_id}.pt")
        torch.save(save_data, save_path)

        eeg_time = time.time() - eeg_start
        print(f"[{i+1}/{len(dataset)}] ✅ {eeg_id} — {len(labels)} windows — {eeg_time:.2f}s", end='\r')

    except Exception as e:
        print(f"❌ Failed to save {eeg_id}: {e}")

total_time = time.time() - total_start
print(f"\n\n✅ All sequences saved to: {save_folder}")
print(f"⏱️ Total time: {total_time:.2f} seconds")