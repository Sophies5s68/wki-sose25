import os
import torch
import numpy as np
from wettbewerb import EEGDataset
from new_preprocess import preprocess_signal_with_montages
from new_features import window_eeg_data
from faster_features import compute_spectrogram
import time

# === Config ===
window_size = 4  # seconds
step_size = 2    # seconds
target_fs = 256  # Hz
save_folder = f"raw_dataset/sequences_spectrograms/win{window_size}_step{step_size}"
os.makedirs(save_folder, exist_ok=True)

# Spectrogram params
n_fft = 64
hop_length = 32

def compute_spectrogram(window, sample_rate=target_fs, n_fft=n_fft, hop_length=hop_length):
    """
    Compute magnitude spectrogram using STFT per EEG channel.
    Input: window [channels, samples]
    Output: spec [channels, freq_bins, time_bins]
    """
    window_tensor = torch.tensor(window, dtype=torch.float32)  # [C, S]
    specs = []

    for ch in window_tensor:
        stft = torch.stft(
            ch,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft),
            return_complex=True  # get complex STFT
        )
        magnitude = stft.abs()
        log_spec = torch.log1p(magnitude)# take magnitude only
        specs.append(log_spec)  # [freq_bins, time_bins]

    return torch.stack(specs)  # [channels, freq_bins, time_bins]                 # shape [channels, freq_bins, time_bins]

# Load EEG dataset
dataset = EEGDataset("../shared_data/training")
start_time = time.time()

for i in range(len(dataset)):
    eeg_id, channels, raw_data, fs, _, label = dataset[i]
    seizure_label, seizure_onset, seizure_offset = label

    montage_names, processed_signal, montage_missing, new_fs = preprocess_signal_with_montages(
        channels, raw_data, target_fs=target_fs, original_fs=fs, ids=eeg_id
    )
    if montage_missing:
        print(f"Skipping {eeg_id} (montage missing)")
        continue

    windows, labels, timestamps, used_whole_recording = window_eeg_data(
        processed_signal,
        resampled_fs=new_fs,
        seizure_onset=seizure_onset,
        seizure_offset=seizure_offset,
        window_size=window_size,
        step_size=step_size
    )
    if used_whole_recording:
        print(f"⚠️ Used whole recording as one window for patient {eeg_id}")

    spectrogram_windows = []
    for window in windows:
        spec = compute_spectrogram(window)
        spec_norm = (spec - spec.mean(dim=(1,2), keepdim=True)) / (spec.std(dim=(1,2), keepdim=True) + 1e-8)
        spectrogram_windows.append(spec_norm)
    spectrogram_tensor = torch.stack(spectrogram_windows)

    # Sequence label = seizure present in recording (can also use seizure_label directly)
    sequence_label = seizure_label

    save_path = os.path.join(save_folder, f"{eeg_id}_seq.pt")
    torch.save({
        "windows": spectrogram_tensor,        # List of tensors [channels, freq_bins, time_bins]
        "label": sequence_label,
        "window_labels": torch.tensor(labels, dtype=torch.long),
        "eeg_id": eeg_id,
        "timestamps": timestamps,
        "seizure_onset": seizure_onset,
        "seizure_offset": seizure_offset
    }, save_path)

    print(f"[{i+1}/{len(dataset)}] Saved {eeg_id} with {len(windows)} spectrogram windows.", end='\r')

print(f"\nDone. Time elapsed: {time.time() - start_time:.2f} seconds")