# Erstellen eines einfachen Datasets:
# Output folder, window und step size angeben, wird direkt an CNN_dataset übergeben 

'''
import CNN_dataset
from wettbewerb import load_references, get_3montages
import os

train_folder = "../shared_data/training"
output_folder = "data_test"
window_size = 30
step_size = 30


os.makedirs(output_folder, exist_ok=True)
files = [f for f in os.listdir(train_folder) if f.endswith('.mat')]
n_files = len(files)
print(f"found {n_files} files")

index = 0
for i in range(0, n_files, 100):
    ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(train_folder, i)
    CNN_dataset.create_cnn_dataset_map(ids, channels, data, sampling_frequencies, reference_systems, eeg_labels,window_size,step_size, output_folder, i)
    print(f"created dataset {index}")
    index = index + 1
'''  
    
    
    
    
# Erstellen mehrerer Datensätze automatisiert
# window configs angeben, in Ordner datasets werden so benannte Unterordner erstellt

import CNN_dataset
from wettbewerb import load_references, get_3montages
import os

train_folder = "../shared_data/training"
files = [f for f in os.listdir(train_folder) if f.endswith('.mat')]
n_files = len(files)

print(f"found {n_files} files")
window_configs = [(5,5),(10,10),(10,5),(20,20),(20,10),(30,30),(30,15),(60,60),(60,30)]
index = 0
for i in range(0, n_files, 100):
    ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(train_folder, i)
    CNN_dataset.create_multiple_cnn_datasets(ids, channels, data, sampling_frequencies, reference_systems, eeg_labels,index,window_configs, base_output_dir="datasets")
    print(f"\n created datasets {index}")
    index = index + 1
    
    
