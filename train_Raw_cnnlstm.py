import os
import torch
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

# === CONFIG ===
dataset_folder = "raw_dataset/win4_step1"
n_splits = 5
random_state = 42

# === Step 1: Load windows and group by patient ===
patient_to_windows = defaultdict(list)

for fname in os.listdir(dataset_folder):
    if fname.endswith(".pt"):
        path = os.path.join(dataset_folder, fname)
        features, label, patient_id, timestamp = torch.load(path)
        patient_to_windows[patient_id].append((features, label))

print(f"✅ Loaded data from {len(patient_to_windows)} unique patients.")

# === Step 2: Assign binary label to each patient (1 = any seizure) ===
patient_ids = list(patient_to_windows.keys())
patient_labels = np.array([
    int(any(lbl == 1 for _, lbl in patient_to_windows[pid]))
    for pid in patient_ids
])

print(f"Label distribution across patients: {np.bincount(patient_labels)}")

# === Step 3: Stratified K-Fold Split ===
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
    print(f"\n🧪 Fold {fold + 1}")

    train_pids = [patient_ids[i] for i in train_idx]
    test_pids = [patient_ids[i] for i in test_idx]

    # === Step 4: Flatten window data for model training ===
    train_data = [sample for pid in train_pids for sample in patient_to_windows[pid]]
    test_data = [sample for pid in test_pids for sample in patient_to_windows[pid]]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    X_train = np.stack(X_train)
    y_train = np.array(y_train)
    X_test = np.stack(X_test)
    y_test = np.array(y_test)

    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")
    print(f"Train label dist: {np.bincount(y_train)} | Test label dist: {np.bincount(y_test)}")

    # ⬇️ You can now use X_train, y_train, X_test, y_test for training your model

    break  # remove this to loop through all folds

import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Wrap in DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)