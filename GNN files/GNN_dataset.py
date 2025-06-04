import torch
from torch_geometric.data import Data
import numpy as np

from preprocess import process_eeg, process_without_mne
from features import feature_extraction

# Definiere ein festes Set an EEG channels für Graphbuilding
standard_channels = [ 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'T3', 'T4']
n_nodes = len(standard_channels)

# Definiere den Graph (fully connected für jedes recording gleich)
edge_index = torch.tensor([[i,j] for i in range (n_nodes) for j in range(n_nodes) if i!=j],dtype = torch.long).t().contiguous()

# das sollten wir als Funktion machen, sonst wird das beim importieren ins Notebook ausgeführt wird
# EEG Data laden

window_size = 4.0
step_size = 1


#das loaod references würd ich dann einfach im Dataloader aufrufen und die ergebnisse übergeben
#ids, channels_list, data_list, fs_list, ref_list, label_list = load_references("../shared_data/training_mini")
def create_graphs(ids,channels_list,data_list,fs_list,ref_list,label_list):
    print("create_graphs() was called")
    print(f"label_list type: {type(label_list)}")
    print(f"label_list[0]: {label_list[0]}")
    graph_list = []
    
    lowest_sampling = np.min(fs_list)
    for i in range (len(ids)):
        print(f"\n--- Graph {i} ---")
        print(f"ID: {ids[i]}")
        print(f"Label tuple: {label_list[i]}")
        label = label_list[i][0]
        label_onset = label_list[i][1]
        label_offset = label_list[i][2]
        print(f"Assigned label: {label} ({int(label)})")
        channels = channels_list[i]
        data = data_list[i]
        fs = fs_list[i]
        ref = ref_list[i]
        label = int(label)
        
        #clean_data = process_eeg(data, fs, channels, ref)
        clean_data = process_without_mne(data,fs,channels,ref,lowest_sampling)
        windows, labels_window = window_data(clean_data,fs,window_size,step_size,label,label_onset,label_offset)
        
         # Referenziere Channel zu Index 
        channel_map = {ch: idx for idx, ch in enumerate(channels)}
        
        pad_data = np.zeros((n_nodes, int(window_size*fs)))
        for index, window in enumerate(windows):
    
            # Falls fehlende Channels: Zero-Padding
            for j, ch in enumerate(standard_channels):
                if ch in channel_map:
                    pad_data[j] = window[channel_map[ch]]
            # Extract features per node
            node_features = feature_extraction(pad_data, fs) # solle shape von n_features haben
            x = torch.tensor(node_features, dtype=torch.float) # shape: [n_nodes, n_features]
            y = torch.tensor(labels_window[index], dtype= torch.long) # Graph label
    
            # Build graph
            graph = Data(x=x, edge_index=edge_index, y=y)
            #print(f"Processed: {ids[i]}, window {index}")
            graph_list.append(graph)
    
    torch.save(graph_list, "dataset.pt")
    print("Dataset mit Graphen wurde erstellt")
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
            
    
    
    