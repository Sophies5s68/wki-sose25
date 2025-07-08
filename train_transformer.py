# === Seizure Detection + Onset Prediction Pipeline ===

import os
import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import WeightedRandomSampler
from Transformer_Model import CNNTransformer_1D, CNNTransformer_2D

# === CONFIG ===
dataset_folder = "raw_dataset/sequences_spectrograms/win4_step2"
n_splits = 5
random_state = 42
batch_size = 16
num_epochs = 50
step_size = 2  # seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# === Load patient files and labels ===
patient_to_path = {}
patient_to_label = {}

file_list = [f for f in os.listdir(dataset_folder) if f.endswith(".pt")]
for i, fname in enumerate(file_list):
    path = os.path.join(dataset_folder, fname)
    data = torch.load(path, map_location='cpu')
    patient_id = data["eeg_id"]
    patient_to_path[patient_id] = path
    patient_to_label[patient_id] = data["label"]
    print(f"Loaded {i+1}/{len(file_list)}: {fname}", end='\r')

print(f"\nLoaded {len(patient_to_path)} patient files.")

# === Stratified Split ===
patient_ids = list(patient_to_path.keys())
patient_labels = np.array([patient_to_label[pid] for pid in patient_ids])

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
    train_ids = [patient_ids[i] for i in train_idx]
    test_ids = [patient_ids[i] for i in test_idx]
    print(f"\nFold {fold+1} | Train: {len(train_ids)} | Test: {len(test_ids)}")
    break
num_seizure_patients = sum([1 for pid in train_ids if patient_to_label[pid] == 1])
print("Seizure patients in training set:", num_seizure_patients)

# === Datasets ===
class EEGSequenceDataset(Dataset):
    def __init__(self, ids, paths, labels, onset_mode=False, chunk_len=10, stride=10, pad_short_sequences=False):
        self.samples = []
        self.onset_mode = onset_mode
        self.chunk_len = chunk_len
        self.pad_short_sequences = pad_short_sequences

        for pid in ids: 
            data = torch.load(paths[pid], map_location='cpu')
            windows = data["windows"]
            total_len = windows.size(0)

            if onset_mode:
                window_labels = data["window_labels"]
            if total_len < chunk_len: 
                if pad_short_sequences and onset_mode: 
                    pad_len = chunk_len - total_len
                    windows = torch.cat([windows, torch.zeros((pad_len, *windows.shape[1:]))], dim=0)
                    window_labels = torch.cat([window_labels, torch.full((pad_len,), -100)])
                    self.samples.append((windows, window_labels))
                else:
                    continue
            else: 
                for i in range(0, total_len - chunk_len + 1, stride):
                    chunk = windows[i:i+chunk_len]
                    if onset_mode:
                        chunk_labels = window_labels[i:i+chunk_len]
                        self.samples.append((chunk, chunk_labels))
                    else:
                        chunk_window_labels = data["window_labels"][i:i+chunk_len]
                        seizure_label = 1 if (chunk_window_labels == 1).any() else 0
                        self.samples.append((chunk, seizure_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# === Collate Functions ===
def collate_fn_detection(batch):
    sequences, labels = zip(*batch)
    padded_seq = pad_sequence(sequences, batch_first=True)
    return padded_seq, torch.tensor(labels, dtype=torch.long)

def collate_fn_onset(batch):
    sequences, labels = zip(*batch)
    padded_seq = pad_sequence(sequences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_seq, padded_labels

# === Dataloaders ===
train_dataset_det = EEGSequenceDataset(train_ids, patient_to_path, patient_to_label, onset_mode=False)
test_dataset_det = EEGSequenceDataset(test_ids, patient_to_path, patient_to_label, onset_mode=False)


# Get labels for all chunks
chunk_labels = [label for _, label in train_dataset_det]
# Assign weights inversely proportional to class frequency
label_counts = Counter(chunk_labels)
weights = [1.0 / label_counts[label] for label in chunk_labels]
# Create sampler
sampler = WeightedRandomSampler(weights, num_samples=len(chunk_labels), replacement=True)
# Use the sampler in DataLoader
train_loader_det = DataLoader(train_dataset_det, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn_detection)
test_loader_det = DataLoader(test_dataset_det, batch_size=batch_size, collate_fn=collate_fn_detection)

counter = Counter([label for _, label in train_dataset_det])
print("Training chunk class distribution:", counter)

sample_input, _ = next(iter(train_loader_det))

# === Stage 1: Seizure Detection Model ===
detector = CNNTransformer_2D(
    num_channels=sample_input.shape[2],
    freq_bins=sample_input.shape[3],
    time_bins=sample_input.shape[4],
    num_classes=2,
    per_window=False
).to(device)

counts = Counter([patient_to_label[pid] for pid in train_ids])
total = counts[0] + counts[1]
weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float32).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)

print("\n=== Stage 1: Training Seizure Detection ===")
for epoch in range(1, num_epochs + 1):
    detector.train()
    for X, y in train_loader_det:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = detector(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    detector.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader_det:
            batch_counts = Counter(y.cpu().numpy())
            X = X.to(device)
            out = detector(X)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    f1 = f1_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Epoch {epoch} | F1: {f1:.4f} | Acc: {acc:.4f}")
    print("Predicted class counts:", Counter(all_preds))
    print("True class counts:", Counter(all_targets))

# === Stage 2: Onset Prediction (only on seizure-positive samples) ===
print("\n=== Stage 2: Training Onset Prediction ===")
seizure_train_ids = [pid for pid in train_ids if patient_to_label[pid] == 1]
seizure_test_ids = [pid for pid in test_ids if patient_to_label[pid] == 1]

train_dataset_onset = EEGSequenceDataset(seizure_train_ids, patient_to_path, None, onset_mode=True)
test_dataset_onset = EEGSequenceDataset(seizure_test_ids, patient_to_path, None, onset_mode=True)

train_loader_onset = DataLoader(train_dataset_onset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_onset)
test_loader_onset = DataLoader(test_dataset_onset, batch_size=batch_size, collate_fn=collate_fn_onset)

onset_model = CNNTransformer_2D(
    num_channels=sample_input.shape[2],
    freq_bins=sample_input.shape[3],
    time_bins=sample_input.shape[4],
    num_classes=2,
    per_window=True
).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
onset_optimizer = torch.optim.Adam(onset_model.parameters(), lr=1e-3)

for epoch in range(1, num_epochs + 1):
    onset_model.train()
    for X, y in train_loader_onset:
        X, y = X.to(device), y.to(device)
        onset_optimizer.zero_grad()
        out = onset_model(X)  # [B, T, 2]
        loss = criterion(out.view(-1, 2), y.view(-1))
        loss.backward()
        onset_optimizer.step()

    onset_model.eval()
    all_preds, all_targets = [], []
    onset_errors = []
    with torch.no_grad():
        for X, y in test_loader_onset:
            X, y = X.to(device), y.to(device)
            out = onset_model(X)
            preds = out.argmax(dim=-1)  # [B, T]
            mask = y != -100

            for b in range(X.size(0)):
                pred_seq = preds[b]
                true_seq = y[b]
                valid_mask = mask[b]

                pred_idx = (pred_seq[valid_mask] == 1).float().argmax().item() if (pred_seq[valid_mask] == 1).any() else -1
                true_idx = (true_seq[valid_mask] == 1).float().argmax().item() if (true_seq[valid_mask] == 1).any() else -1

                pred_time = pred_idx * step_size if pred_idx >= 0 else "None"
                true_time = true_idx * step_size if true_idx >= 0 else "None"

                if isinstance(pred_time, (int, float)) and isinstance(true_time, (int, float)):
                    onset_errors.append(abs(pred_time - true_time))

            all_preds.extend(preds[mask].cpu().numpy())
            all_targets.extend(y[mask].cpu().numpy())

    f1 = f1_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Epoch {epoch} | Onset F1: {f1:.4f} | Onset Acc: {acc:.4f}")
    if onset_errors:
        print(f"Mean Onset Error: {np.mean(onset_errors):.2f} sec")