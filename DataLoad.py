import CNN_dataset
from wettbewerb import load_references, get_3montages
import os

train_folder = "../shared_data/training"
files = [f for f in os.listdir(train_folder) if f.endswith('.mat')]
n_files = len(files)
print(f"found {n_files} files")

index = 0
for i in range(0, n_files, 100):
    ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(train_folder, i)
    CNN_dataset.create_cnn_dataset_map(ids, channels, data, sampling_frequencies, reference_systems, eeg_labels, i)
    print(f"created dataset {index}")
    index = index + 1