import torch
import numpy as np

from preprocess import process_eeg, process_without_mne
from features import feature_extraction
from scipy import signal
import os

# Definiere ein festes Set an EEG channels für Graphbuilding
standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
n_nodes = len(standard_channels)


def create_cnn_dataset(ids, channels_list, data_list, fs_list, ref_list, label_list, index):

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
    #torch.save(dataset, f"data/cnn_dataset_{index}.pt") ## NICHT HIER
    return dataset

def create_cnn_dataset_map(ids, channels_list, data_list, fs_list, ref_list, label_list,window_size, step_size, output_folder,index=0):

    dataset = []
    lowest_sampling = np.min(fs_list)

    for i in range(len(ids)):
        #print(f"\n--- Subject {i} --- ID: {ids[i]}")
        channels = channels_list[i]
        data = data_list[i]
        fs = fs_list[i]
        ref = ref_list[i]
        label, onset, offset = label_list[i]
        label = int(label)

        # Preprocessing
        clean_data = process_without_mne(data, fs, channels, ref, lowest_sampling)
        assert not np.isnan(clean_data).any(), "NaN in clean_data!"

        # Channel-Mapping (Standardisierung auf gleiche Reihenfolge und Shape)
        channel_map = {ch: idx for idx, ch in enumerate(channels)}
        pad_data = np.zeros((n_nodes, clean_data.shape[1]))
        for j, ch in enumerate(standard_channels):
            if ch in channel_map:
                pad_data[j] = clean_data[channel_map[ch]]
            assert not np.isnan(pad_data).any(), "NaN in pad_data!"
        '''
        # Aufteilen in Bänder für Feature processing
        alpha_data = np.zeros(pad_data.shape, dtype=complex)
        theta_data = np.zeros(pad_data.shape, dtype=complex)
        gamma_data = np.zeros(pad_data.shape, dtype=complex)
        for index, dat in enumerate(pad_data):
            alpha = bandpass_filter(dat,lowest_sampling,4,8)
            alpha_data[index] = signal.hilbert(alpha)
            theta = bandpass_filter(dat,lowest_sampling,1,4)
            theta_data[index] = signal.hilbert(theta)
            gamma = bandpass_filter(dat,lowest_sampling,30,120)
            gamma_data[index] = signal.hilbert(gamma)
        ''' 
        # Fenster extrahieren
        windows, window_labels = window_data(pad_data, fs, window_size, step_size, label, onset, offset)
        #alpha_windows = window_nolabel(alpha_data,fs,window_size,step_size)
        #theta_windows = window_nolabel(theta_data,fs,window_size,step_size)
        #gamma_windows = window_nolabel(gamma_data,fs,window_size,step_size)
        
        # Feature extraction and brain map calculation
        #for w, l, a, t, g in zip(windows,window_labels,alpha_windows,theta_windows,gamma_windows):
        
        for w, l in zip(windows,window_labels):
            features = feature_extraction(w, lowest_sampling) # shape: (n_channels, n_features)
            assert not np.isnan(features).any(), "NaN in features!"
            brain_map = create_fixed_grid_maps(features,channels)
            assert not np.isnan(brain_map).any(), "NaN in brain_map!"
            x = torch.tensor(brain_map, dtype = torch.float)
            y = torch.tensor(l, dtype = torch.long)
            patient_id = ids[i].split("_")[0]
            dataset.append((x,y,patient_id))
            
    save_path = os.path.join(output_folder, f"cnn_map_dataset_{index}.pt")
    torch.save(dataset, save_path)
    print("Dataset mit Maps erstellt")
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

def window_nolabel(data,fs,window_size_sec,step_size_sec):
    window_size = int(window_size_sec * fs)
    step_size = int(step_size_sec * fs)
    n_samples = data.shape[1]
    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        stop = start + window_size
        window = data[:, start:stop]
        windows.append(window)
    return windows

def create_fixed_grid_maps(features, channels):
    layout = [
    [0,   'Fp1',  0,   'Fp2',  0],
    ['F7', 'F3', 'Fz', 'F4', 'F8'],
    ['T3', 'C3', 'Cz', 'C4', 'T4'],
    ['T5', 'P3', 'Pz', 'P4', 'T6'],
    [0,   'O1',  0,   'O2',  0]]
    
    channel_idx = {ch: i for i, ch in enumerate(channels)}
    
    n_features = features.shape[1]
    H, W = len(layout), len(layout[0])
    brain_maps = np.zeros((n_features, H, W))
    
    for i_feat in range(n_features):
        for i in range(H):
            for j in range(W):
                ch = layout[i][j]
                if ch != 0 and ch in channel_idx:
                    ch_index = channel_idx[ch]
                    brain_maps[i_feat, i, j] = features[ch_index, i_feat]
    return brain_maps
                    
    
def window_data_evaluate(data, fs, window_size_sec, step_size_sec):
    window_size = int(window_size_sec * fs)
    step_size = int(step_size_sec * fs)
    n_samples = data.shape[1]
    
    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        stop = start + window_size
        window = data[:, start:stop]
        windows.append(window)
    return windows

# Test das Filtern hierher zu verlegen für bessere Laufzeit
def bandpass_filter(sig, fs, lowcut, highcut,numtaps=101, window='hamming'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    taps = signal.firwin(numtaps, [low, high], pass_zero=False, window=window)
    return signal.lfilter(taps,1.0,sig)


def create_multiple_cnn_datasets(
    ids, channels_list, data_list, fs_list, ref_list, label_list, index,
    window_configs,  # Liste von (window_size, step_size)
    base_output_dir="cnn_datasets"
):
    os.makedirs(base_output_dir, exist_ok=True)
    lowest_sampling = np.min(fs_list)
    preprocessed_data = []  # Liste von Tupeln: (pad_data, channels, label, onset, offset, patient_id)


    for i in range(len(ids)):
        channels = channels_list[i]
        data = data_list[i]
        fs = fs_list[i]
        ref = ref_list[i]
        label, onset, offset = label_list[i]
        label = int(label)
        patient_id = ids[i].split("_")[0]

        clean_data = process_without_mne(data, fs, channels, ref, lowest_sampling)
        assert not np.isnan(clean_data).any(), f"NaN in clean_data für {patient_id}"

        channel_map = {ch: idx for idx, ch in enumerate(channels)}
        pad_data = np.zeros((n_nodes, clean_data.shape[1]))
        for j, ch in enumerate(standard_channels):
            if ch in channel_map:
                pad_data[j] = clean_data[channel_map[ch]]
        assert not np.isnan(pad_data).any(), f"NaN in pad_data für {patient_id}"
        
            
        preprocessed_data.append((pad_data, channels, label, onset, offset, patient_id))

    for window_size, step_size in window_configs:
        output_dir = os.path.join(base_output_dir, f"win{window_size}_step{step_size}")
        os.makedirs(output_dir, exist_ok=True)
        dataset = []
        for pad_data, channels, label, onset, offset, patient_id in preprocessed_data:
            windows, window_labels = window_data(pad_data, lowest_sampling, window_size, step_size,label, onset, offset)
            
            for w, l in zip(windows,window_labels):
                features = feature_extraction(w, lowest_sampling)
                assert not np.isnan(features).any(), f"NaN in features für {patient_id}"
                brain_map = create_fixed_grid_maps(features, channels)
                assert not np.isnan(brain_map).any(), f"NaN in brain_map für {patient_id}"

                x = torch.tensor(brain_map, dtype=torch.float)
                y = torch.tensor(l, dtype=torch.long)
                dataset.append((x, y, patient_id))

        save_path = os.path.join(output_dir, f"cnn_map_dataset_{index}.pt" )
        torch.save(dataset, save_path)
        print(f" Gespeichert: {save_path}")

        
def create_multiple_cnn_datasets_plv(
    ids, channels_list, data_list, fs_list, ref_list, label_list, index,
    window_configs,  # Liste von (window_size, step_size)
    base_output_dir="cnn_datasets"
):
    os.makedirs(base_output_dir, exist_ok=True)
    lowest_sampling = np.min(fs_list)
    preprocessed_data = []  # Liste von Tupeln: (pad_data, channels, label, onset, offset, patient_id)


    for i in range(len(ids)):
        channels = channels_list[i]
        data = data_list[i]
        fs = fs_list[i]
        ref = ref_list[i]
        label, onset, offset = label_list[i]
        label = int(label)
        patient_id = ids[i].split("_")[0]

        clean_data = process_without_mne(data, fs, channels, ref, lowest_sampling)
        assert not np.isnan(clean_data).any(), f"NaN in clean_data für {patient_id}"

        channel_map = {ch: idx for idx, ch in enumerate(channels)}
        pad_data = np.zeros((n_nodes, clean_data.shape[1]))
        for j, ch in enumerate(standard_channels):
            if ch in channel_map:
                pad_data[j] = clean_data[channel_map[ch]]
        assert not np.isnan(pad_data).any(), f"NaN in pad_data für {patient_id}"
        
        # Aufteilen in Bänder für Feature processing
        alpha_data = np.zeros(pad_data.shape, dtype=complex)
        theta_data = np.zeros(pad_data.shape, dtype=complex)
        gamma_data = np.zeros(pad_data.shape, dtype=complex)
        for index, dat in enumerate(pad_data):
            alpha = bandpass_filter(dat,lowest_sampling,4,8)
            alpha_data[index] = signal.hilbert(alpha)
            theta = bandpass_filter(dat,lowest_sampling,1,4)
            theta_data[index] = signal.hilbert(theta)
            gamma = bandpass_filter(dat,lowest_sampling,30,120)
            gamma_data[index] = signal.hilbert(gamma)
            
        preprocessed_data.append((pad_data, channels, label, onset, offset, patient_id, alpha_data, theta_data, gamma_data))

    for window_size, step_size in window_configs:
        output_dir = os.path.join(base_output_dir, f"win{window_size}_step{step_size}")
        os.makedirs(output_dir, exist_ok=True)
        dataset = []
        for pad_data, channels, label, onset, offset, patient_id, alpha_data, theta_data, gamma_data in preprocessed_data:
            windows, window_labels = window_data(pad_data, lowest_sampling, window_size, step_size,label, onset, offset)
            alpha_windows = window_nolabel(alpha_data,fs,window_size,step_size)
            theta_windows = window_nolabel(theta_data,fs,window_size,step_size)
            gamma_windows = window_nolabel(gamma_data,fs,window_size,step_size)
            for w, l, a, t, g in zip(windows,window_labels,alpha_windows,theta_windows,gamma_windows):
                features = feature_extraction(w, lowest_sampling, a, t, g)
                assert not np.isnan(features).any(), f"NaN in features für {patient_id}"
                brain_map = create_fixed_grid_maps(features, channels)
                assert not np.isnan(brain_map).any(), f"NaN in brain_map für {patient_id}"

                x = torch.tensor(brain_map, dtype=torch.float)
                y = torch.tensor(l, dtype=torch.long)
                dataset.append((x, y, patient_id))

        save_path = os.path.join(output_dir, f"cnn_map_dataset_{index}.pt" )
        torch.save(dataset, save_path)
        print(f" Gespeichert: {save_path}")
