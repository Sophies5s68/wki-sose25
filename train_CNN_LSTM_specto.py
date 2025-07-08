import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from CNNLSTMcomb import CNN_LSTM_2D

# === CONFIG ===
dataset_folder = "raw_dataset/raw_sequences_spectrograms/win2_step2"
n_splits = 5
random_state = 42
batch_size = 4
num_epochs = 50

# === Load paths and labels only ===
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

# === Dataset Class ===
class EEGSequenceDataset(Dataset):
    def __init__(self, patient_ids, patient_paths, patient_labels):
        self.ids = patient_ids
        self.paths = patient_paths
        self.labels = patient_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        data = torch.load(self.paths[pid], map_location='cpu')
        windows = torch.stack(data["windows"])  # [T, C, F, S]
        label = self.labels[pid]
        return windows, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seq = pad_sequence(sequences, batch_first=True)  # [B, T, C, F, S]
    return padded_seq, torch.tensor(labels, dtype=torch.long)

train_dataset = EEGSequenceDataset(train_ids, patient_to_path, patient_to_label)
test_dataset = EEGSequenceDataset(test_ids, patient_to_path, patient_to_label)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_input, _ = next(iter(train_loader))
model = CNN_LSTM_2D(
    num_channels=sample_input.shape[2],
    freq_bins=sample_input.shape[3],
    time_bins=sample_input.shape[4],
    hidden_dim=16,
    num_classes=2
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

# === Training Loop ===
for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses, train_preds, train_targets = [], [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        with autocast():
            output = model(X_batch)
            loss = criterion(output, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

        train_losses.append(loss.item())
        train_preds.extend(output.argmax(1).cpu().numpy())
        train_targets.extend(y_batch.cpu().numpy())

    train_f1 = f1_score(train_targets, train_preds)
    train_acc = accuracy_score(train_targets, train_preds)

    model.eval()
    test_losses, test_preds, test_targets = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with autocast():
                output = model(X_batch)
                loss = criterion(output, y_batch)
            test_losses.append(loss