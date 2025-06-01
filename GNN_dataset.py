import torch
from torch_geometric.data import Data
import numpy as np

from preprocess import process_eeg
from features import feature_extraction

# Definiere ein festes Set an EEG channels für Graphbuilding
standard_channels = [ 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'T3', 'T4']
n_nodes = len(standard_channels)

# Definiere den Graph (fully connected für jedes recording gleich)
edge_index = torch.tensor([[i,j] for i in range (n_nodes) for j in range(n_nodes) if i!=j],dtype = torch.long).t().contiguous()

# das sollten wir als Funktion machen, sonst wird das beim importieren ins Notebook ausgeführt wird
# EEG Data laden

#das loaod references würd ich dann einfach im Dataloader aufrufen und die ergebnisse übergeben
#ids, channels_list, data_list, fs_list, ref_list, label_list = load_references("../shared_data/training_mini")
def create_graphs(ids,channels_list,data_list,fs_list,ref_list,label_list):
    print("create_graphs() was called")
    print(f"label_list type: {type(label_list)}")
    print(f"label_list[0]: {label_list[0]}")
    graph_list = []
    for i in range (len(ids)):
        print(f"\n--- Graph {i} ---")
        print(f"ID: {ids[i]}")
        print(f"Label tuple: {label_list[i]}")
        label = label_list[i][0]
        print(f"Assigned label: {label} ({int(label)})")
        channels = channels_list[i]
        data = data_list[i]
        fs = fs_list[i]
        ref = ref_list[i]
        label = int(label)
    
        # Preprocess EEG data [n_channels, n_samples]
        clean_data = process_eeg(data, fs, channels, ref)
    
        # Referenziere Channel zu Index 
        channel_map = {ch: idx for idx, ch in enumerate(channels)}
        n_samples= clean_data.shape[1]
    
        # Falls fehlende Channels: Zero-Padding
        pad_data = np.zeros((n_nodes, n_samples))
        for j, ch in enumerate(standard_channels):
            if ch in channel_map:
                pad_data[j] = clean_data[channel_map[ch]]
        # Extract features per node
        node_features = feature_extraction(pad_data, fs) # solle shape von n_features haben
        x = torch.tensor(node_features, dtype=torch.float) # shape: [n_nodes, n_features]
        y= torch.tensor(label, dtype= torch.long) # Graph label
    
        # Build graph
        graph = Data(x=x, edge_index=edge_index, y=y)
        print(f"Processed: {ids[i]}")
        graph_list.append(graph)
    
    torch.save(graph_list, "dataset.pt")
    print("Dataset mit Graphen wurde erstellt")
    return graph_list
    
    
    
    
            
    
    
    