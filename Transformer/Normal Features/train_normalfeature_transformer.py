import torch
import os
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from CNN_Transformer import CNNTransformerEEG

# === EEG Dataset Class ===
class EEGSequenceDataset(Dataset):
    def __init__(self, ids, patient_to_path, labels=None, onset_mode=False,
                 chunk_len=10, stride=10, cache_data=False):
        self.onset_mode = onset_mode
        self.chunk_len = chunk_len
        self.paths = patient_to_path
        self.sample_metadata = []
        self.chunk_labels = []
        self.cache_data = cache_data
        self.data_cache = {} if cache_data else None

        for pid in ids:
            data = torch.load(self.paths[pid], map_location='cpu')
            windows = data["windows"]
            window_labels = data["window_labels"]
            T = windows.size(0)

            for i in range(0, T - chunk_len + 1, stride):
                self.sample_metadata.append({"pid": pid, "start": i})
                if not onset_mode:
                    chunk = window_labels[i:i + chunk_len]
                    seizure_label = 1 if (chunk.sum() >= 1) else 0
                    self.chunk_labels.append(seizure_label)
            if cache_data:
                self.data_cache[pid] = data

    def __len__(self):
        return len(self.sample_metadata)

    def __getitem__(self, idx):
        meta = self.sample_metadata[idx]
        pid = meta["pid"]
        start = meta["start"]

        data = self.data_cache[pid] if self.cache_data else torch.load(self.paths[pid], map_location='cpu')
        x = data["windows"][start:start + self.chunk_len]

        if self.onset_mode:
            y = data["window_labels"][start:start + self.chunk_len]
            return x, y
        else:
            y = self.chunk_labels[idx]
            return x, y

# === Collate Functions ===
def collate_fn_detection(batch):
    sequences, labels = zip(*batch)
    padded_seq = pad_sequence(sequences, batch_first=True)
    return padded_seq, torch.tensor(labels, dtype=torch.long)

def collate_fn_onset(batch):
    sequences, labels = zip(*batch)
    padded_seq = pad_sequence(sequences, batch_first=True)
    padded_lbls = pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_seq, padded_lbls

# === Config ===
data_folder = "data_new_window/win4_step1_transformer"
n_splits = 5
batch_size = 16
num_epochs = 50
chunk_len = 10
stride = 5

csv_detection_path = "models_newWin/Transformer/metrics_detection_allfolds.csv"
csv_onset_path = "models_newWin/Transformer/metrics_onset_allfolds.csv"

# Ensure output directories exist
os.makedirs(os.path.dirname(csv_detection_path), exist_ok=True)
os.makedirs(os.path.dirname(csv_onset_path), exist_ok=True)

# Write CSV headers if not exists
if not os.path.exists(csv_detection_path):
    with open(csv_detection_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "epoch", "train_loss", "train_acc", "val_acc", "f1_score"])

if not os.path.exists(csv_onset_path):
    with open(csv_onset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "epoch", "f1", "acc", "mean_onset_error"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on", device)

# === Load Paths and Labels ===
all_files = glob.glob(f"{data_folder}/*.pt")
patient_to_path = {os.path.basename(f).split(".")[0]: f for f in all_files}
patient_to_label = {}
for pid, path in patient_to_path.items():
    data = torch.load(path, map_location='cpu')
    label = 1 if (data["window_labels"] == 1).any() else 0
    patient_to_label[pid] = label

patient_ids = list(patient_to_path.keys())
labels_array = np.array([patient_to_label[pid] for pid in patient_ids])

# === Train Across Folds ===
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, labels_array)):
    print(f"\n=== Fold {fold} ===")

    train_ids = [patient_ids[i] for i in train_idx]
    test_ids = [patient_ids[i] for i in test_idx]

    # === Stage 1: Detection ===
    train_dataset = EEGSequenceDataset(train_ids, patient_to_path, onset_mode=False, chunk_len=chunk_len, stride=stride)
    test_dataset = EEGSequenceDataset(test_ids, patient_to_path, onset_mode=False, chunk_len=chunk_len, stride=stride)
    print("Created Train and Test Datasets for Seizure Detection")

    label_counts = Counter(train_dataset.chunk_labels)
    num_pos = label_counts[1]
    num_neg = label_counts[0]
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}

    weights = [class_weights[label] for label in train_dataset.chunk_labels]
    sampler = WeightedRandomSampler(weights, num_samples=int(2 * len(train_dataset)), replacement=True)
    
    print(f"Train seizure ratio: {num_pos / (num_pos + num_neg):.3f}")
    
    num_workers = 4  # or os.cpu_count() // 2
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              collate_fn=collate_fn_detection, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_fn_detection, num_workers=num_workers, pin_memory=pin_memory)

    sample_input, _ = next(iter(train_loader))
    detector = CNNTransformerEEG(
        in_channels=sample_input.shape[2],
        in_features=sample_input.shape[3],
        n_classes=1,
        per_window=False
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Parameters for early stopping
    patience = 5
    no_improve_epochs = 0
    best_f1 = 0.0
    
    print("Start Training for Seizure Detection")
    for epoch in range(1, num_epochs + 1):
        detector.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = detector(X)
            loss = criterion(logits.view(-1), y.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().view(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        detector.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = detector(X)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().view(-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)

        print(f"Fold {fold} | Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")
        with open(csv_detection_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([fold, epoch, avg_loss, train_acc, acc, f1])
        # Early Stopping
        if f1 > best_f1:
            best_f1 = f1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"F1-Score not improving, Early Stopping triggered: {no_improve_epochs}/5")
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch:02d} (no F1 improvement for {patience} epochs)")
                break
        scheduler.step()
    # === Stage 2: Onset ===
    seizure_train_ids = [pid for pid in train_ids if patient_to_label[pid] == 1]
    seizure_test_ids = [pid for pid in test_ids if patient_to_label[pid] == 1]

    train_dataset_onset = EEGSequenceDataset(seizure_train_ids, patient_to_path, onset_mode=True, chunk_len=chunk_len, stride=stride)
    test_dataset_onset = EEGSequenceDataset(seizure_test_ids, patient_to_path, onset_mode=True, chunk_len=chunk_len, stride=stride)
    print("Created Train and Test Datasets for Onset Prediction")
    train_loader_onset = DataLoader(train_dataset_onset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_onset)
    test_loader_onset = DataLoader(test_dataset_onset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_onset)

    # === Model ===
    onset_model = CNNTransformerEEG(
        in_channels=sample_input.shape[2],
        in_features=sample_input.shape[3],
        n_classes=1,
        per_window=True
    ).to(device)

    # === Compute pos_weight for BCEWithLogitsLoss ===
    all_labels = []
    for _, y in train_dataset_onset:
        valid = y[y != -100]
        all_labels.extend(valid.tolist())

    counter = Counter(all_labels)
    pos = counter.get(1, 1)
    neg = counter.get(0, 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(onset_model.parameters(), lr=1e-3)
    print("Start Training for Onset Prediction")
    # === Training Loop ===
    for epoch in range(1, num_epochs + 1):
        onset_model.train()
        for X, y in train_loader_onset:
            X, y = X.to(device), y.to(device)
            mask = y != -100
            y = y.float().unsqueeze(-1)  # (B, T, 1)

            optimizer.zero_grad()
            out = onset_model(X)  # (B, T, 1)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.step()

        # === Evaluation ===
        onset_model.eval()
        all_preds, all_targets, onset_errors = [], [], []

        with torch.no_grad():
            for X, y in test_loader_onset:
                X, y = X.to(device), y.to(device)
                mask = y != -100
                y = y.float().unsqueeze(-1)

                out = onset_model(X)  # (B, T, 1)
                preds = (torch.sigmoid(out) > 0.5).long()  # (B, T, 1)

                for b in range(X.size(0)):
                    pred_seq = preds[b][mask[b]]
                    true_seq = y[b][mask[b]].long().squeeze(-1)

                    if (pred_seq == 1).any() and (true_seq == 1).any():
                        pred_onset = (pred_seq == 1).nonzero(as_tuple=True)[0][0].item()
                        true_onset = (true_seq == 1).nonzero(as_tuple=True)[0][0].item()
                        onset_errors.append(abs(pred_onset - true_onset))

                all_preds.extend(preds[mask].cpu().numpy().flatten())
                all_targets.extend(y[mask].cpu().numpy().flatten())

        f1 = f1_score(all_targets, all_preds)
        acc = accuracy_score(all_targets, all_preds)
        mean_error = np.mean(onset_errors) if onset_errors else -1

        print(f"Fold {fold} | Epoch {epoch:02d} | Onset F1: {f1:.4f} | Acc: {acc:.4f} | Mean Onset Error: {mean_error:.2f}")
        with open(csv_onset_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([fold, epoch, f1, acc, mean_error])