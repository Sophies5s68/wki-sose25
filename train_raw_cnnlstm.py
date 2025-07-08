import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score
from CNNLSTMcomb import CNN_LSTM
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# === CONFIG ===
dataset_folder = "raw_dataset/raw_sequences/win2_step2"
n_splits = 5
random_state = 42
batch_size = 1
num_epochs = 50

# === Step 1: Load file paths and labels only ===
patient_to_path = {}
patient_to_label = {}

file_list = [f for f in os.listdir(dataset_folder) if f.endswith(".pt")]

for i, fname in enumerate(file_list):
    path = os.path.join(dataset_folder, fname)
    data = torch.load(path, map_location='cpu')
    patient_id = data["eeg_id"]
    patient_to_path[patient_id] = path
    patient_to_label[patient_id] = data["label"]
    print(f"Loaded {i + 1}/{len(file_list)}: {fname}", end='\r')

print(f"\n Loaded {len(patient_to_path)} patient files.")

# === Step 2: Stratified Split ===
patient_ids = list(patient_to_path.keys())
patient_labels = np.array([patient_to_label[pid] for pid in patient_ids])

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
    train_ids = [patient_ids[i] for i in train_idx]
    test_ids = [patient_ids[i] for i in test_idx]
    print(f"\n Fold {fold + 1} | Train: {len(train_ids)} | Test: {len(test_ids)}")
    break  # remove this if you want to loop through all folds

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
        windows_list = data["windows"]
        if len(windows_list) == 0:
            print(f"⚠️ Warning: Patient {pid} has 0 windows!")
        windows = torch.stack(windows_list)  # will fail if empty
        label = self.labels[pid]
        return windows, label

# === Collate Function for Padding Sequences ===
def collate_fn(batch):
    sequences, labels = zip(*batch)  # list of [T, C, S]
    padded_seq = pad_sequence(sequences, batch_first=True)  # [B, T, C, S]
    return padded_seq, torch.tensor(labels, dtype=torch.long)

# === Dataloaders ===
train_dataset = EEGSequenceDataset(train_ids, patient_to_path, patient_to_label)
test_dataset = EEGSequenceDataset(test_ids, patient_to_path, patient_to_label)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_input, _ = next(iter(train_loader))
model = CNN_LSTM(
    num_channels=sample_input.shape[2],  # C
    input_length=sample_input.shape[3],
    hidden_dim=16,
    num_classes=2
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Initialize GradScaler for AMP ===
scaler = GradScaler()

# === Training Loop ===
for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses, train_preds, train_targets = [], [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            output = model(X_batch)
            loss = criterion(output, y_batch)

        # Scaled backward pass
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

            # Mixed precision inference
            with autocast():
                output = model(X_batch)
                loss = criterion(output, y_batch)

            test_losses.append(loss.item())
            test_preds.extend(output.argmax(1).cpu().numpy())
            test_targets.extend(y_batch.cpu().numpy())

    test_f1 = f1_score(test_targets, test_preds)
    test_acc = accuracy_score(test_targets, test_preds)

    print(f"\n Epoch {epoch}")
    print(f"Train -> Acc: {train_acc:.4f} | F1: {train_f1:.4f} | Loss: {np.mean(train_losses):.4f}")
    print(f"Test  -> Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Loss: {np.mean(test_losses):.4f}")