import torch
import numpy as np

from preprocess import process_eeg, process_without_mne
from features import feature_extraction

# Definiere ein festes Set an EEG channels für Graphbuilding
standard_channels = [ 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'T3', 'T4']
n_nodes = len(standard_channels)


window_size = 4.0
step_size = 1

def create_cnn_dataset(ids, channels_list, data_list, fs_list, ref_list, label_list):

    dataset = []
    lowest_sampling = np.min(fs_list)

    for i in range(len(ids)):
        print(f"\n--- Subject {i} --- ID: {ids[i]}")
        channels = channels_list[i]
        data = data_list[i]
        fs = fs_list[i]
        ref = ref_list[i]
        label, onset, offset = label_list[i]
        label = int(label)

        # Preprocessing
        clean_data = process_without_mne(data, fs, channels, ref, lowest_sampling)

        # Channel-Mapping (Standardisierung auf gleiche Reihenfolge und Shape)
        channel_map = {ch: idx for idx, ch in enumerate(channels)}
        pad_data = np.zeros((n_nodes, clean_data.shape[1]))
        for j, ch in enumerate(standard_channels):
            if ch in channel_map:
                pad_data[j] = clean_data[channel_map[ch]]

        # Fenster extrahieren
        windows, window_labels = window_data(pad_data, fs, window_size, step_size, label, onset, offset)
        
        
        for w, l in zip(windows, window_labels):
            x = torch.tensor(feature_extraction(w,fs), dtype=torch.float)  # Shape: [n_channels, window_samples]
            y = torch.tensor(l, dtype=torch.long)
            dataset.append((x, y))
    
    print(f"{len(dataset)} windows created")
    torch.save(dataset, "cnn_dataset.pt")
    return 
    
def window_data(data, fs, window_size_sec, step_size_sec,label,label_onset,label_offset):
    window_size = int(window_size_sec * fs)
    step_size = int(step_size_sec * fs)
    n_samples = data.shape[1]
    
    windows = []
    window_labels = []
    for start in range(0, n_samples - window_size + 1, step_size):
        stop = start + window_size
        label_win = 0
        if label == 1:  
            if (start <= label_onset < stop) or (start < label_offset <= stop) or (label_onset <= start and label_offset >= stop):
                label_win = 1  
                
        window_labels.append(label_win)
        window = data[:, start:stop]
        windows.append(window)
    return windows, window_labels