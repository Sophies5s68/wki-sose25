import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def summarize_average_graph_features(dataset_path="dataset.pt", feature_names=None):
    dataset = torch.load(dataset_path)
    
    features_by_label = {0: [], 1: []}
    
    for graph in dataset:
        label = int(graph.y.item())
        node_features = graph.x.numpy()  # shape: [n_nodes, n_features]
        graph_avg = node_features.mean(axis=0)  # → shape: [n_features]
        features_by_label[label].append(graph_avg)
    
    # Compute group averages
    avg_0 = np.mean(features_by_label[0], axis=0)
    avg_1 = np.mean(features_by_label[1], axis=0)

    # Define default feature names
    n_features = avg_0.shape[0]
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"f{i}" for i in range(n_features)]

    # Create and print summary table
    df = pd.DataFrame({
        'Feature': feature_names,
        'No Seizure (0)': avg_0,
        'Seizure (1)': avg_1,
        'Difference': avg_1 - avg_0
    })
    
    pd.set_option('display.precision', 3)


    # Optional: Plot for visual comparison
    x = np.arange(n_features)
    plt.figure(figsize=(12, 5))
    plt.plot(x, avg_0, label='No Seizure (0)', marker='o')
    plt.plot(x, avg_1, label='Seizure (1)', marker='x')
    plt.xticks(x, feature_names, rotation=45)
    plt.title("Average Per-Graph Feature Values")
    plt.ylabel("Standardized Feature Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()