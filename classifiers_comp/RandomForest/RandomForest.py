import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

# === Load Data ===
folder_path = "montage_datasets/combined/win4_step1"
X, y = [], []

for fname in os.listdir(folder_path):
    if fname.endswith(".pt"):
        file_path = os.path.join(folder_path, fname)
        data = torch.load(file_path)
        for sample in data:
            X.append(sample[0].flatten())
            y.append(sample[1])

X = np.array(X)
y = np.array(y)

print("Loaded all .pt files.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label distribution:", np.bincount(y))

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Normalize === (optional for RF, but helps with consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Random Forest ===
print("Training Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print(f"Training complete in {time.time() - start:.2f} seconds.")

# === Evaluation ===
y_train_pred = rf_model.predict(X_train)
y_train_prob = rf_model.predict_proba(X_train)

y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)

# === Metrics ===
metrics = {
    'train_acc': accuracy_score(y_train, y_train_pred),
    'train_f1': f1_score(y_train, y_train_pred),
    'train_loss': log_loss(y_train, y_train_prob),
    'test_acc': accuracy_score(y_test, y_test_pred),
    'test_f1': f1_score(y_test, y_test_pred),
    'test_loss': log_loss(y_test, y_test_prob),
}
df = pd.DataFrame([metrics])
df.to_csv("rf_metrics.csv", index=False)
print("Saved Random Forest metrics to 'rf_metrics.csv'")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_rf.svg")
plt.show()

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix_rf.svg")
plt.show()


# Stratified by Patient ID
print("Starting Model with stratified by patient ID")
# === Load Data Grouped by Patient ===
folder_path = "montage_datasets/combined/win4_step1"
pid_to_samples = defaultdict(list)

for fname in os.listdir(folder_path):
    if fname.endswith(".pt"):
        file_path = os.path.join(folder_path, fname)
        data = torch.load(file_path)
        for sample in data:
            features = sample[0].flatten()
            label = sample[1]
            patient_id = sample[2].split("_")[0]  # e.g., 'aaaaaaac'
            pid_to_samples[patient_id].append((features, label))

# === Create Patient-wise Stratification ===
all_pids = list(pid_to_samples.keys())
pid_labels = [np.bincount([lbl for _, lbl in pid_to_samples[pid]]).argmax() for pid in all_pids]

train_pids, test_pids = train_test_split(
    all_pids, test_size=0.2, stratify=pid_labels, random_state=42
)

X_train, y_train, X_test, y_test = [], [], [], []

for pid in train_pids:
    for features, label in pid_to_samples[pid]:
        X_train.append(features)
        y_train.append(label)

for pid in test_pids:
    for features, label in pid_to_samples[pid]:
        X_test.append(features)
        y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Patient-wise stratified split complete.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Label distribution (train): {np.bincount(y_train)}")
print(f"Label distribution (test): {np.bincount(y_test)}")

# === Normalize === (optional for RF, but for consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Random Forest ===
print("Training Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print(f"Training complete in {time.time() - start:.2f} seconds.")

# === Evaluation ===
y_train_pred = rf_model.predict(X_train)
y_train_prob = rf_model.predict_proba(X_train)

y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)

# === Metrics ===
metrics = {
    'train_acc': accuracy_score(y_train, y_train_pred),
    'train_f1': f1_score(y_train, y_train_pred),
    'train_loss': log_loss(y_train, y_train_prob),
    'test_acc': accuracy_score(y_test, y_test_pred),
    'test_f1': f1_score(y_test, y_test_pred),
    'test_loss': log_loss(y_test, y_test_prob),
}
df = pd.DataFrame([metrics])
df.to_csv("rf_metrics_patient_split.csv", index=False)
print("Saved Random Forest metrics to 'rf_metrics_patient_split.csv'")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (Patient Split)")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_rf_patient_split.svg")


# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Random Forest (Patient Split)")
plt.savefig("confusion_matrix_rf_patient_split.svg")


# === Stratified by Label and Group (Patient) ===
from sklearn.model_selection import StratifiedGroupKFold

print("Starting Model with stratified by label and grouped by patient ID")

# === Load data again, grouped with patient IDs ===
X_all, y_all, pids = [], [], []

for fname in os.listdir(folder_path):
    if fname.endswith(".pt"):
        file_path = os.path.join(folder_path, fname)
        data = torch.load(file_path)
        for sample in data:
            X_all.append(sample[0].flatten())
            y_all.append(sample[1])
            pids.append(sample[2].split("_")[0])  # patient ID

X_all = np.array(X_all)
y_all = np.array(y_all)
pids = np.array(pids)

# === StratifiedGroupKFold ===
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X_all, y_all, groups=pids))

X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

print("Stratified by label and grouped by patient.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Label distribution (train): {np.bincount(y_train)}")
print(f"Label distribution (test): {np.bincount(y_test)}")

# === Normalize ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Train Random Forest ===
print("Training Random Forest (stratified by label & patient)...")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print(f"Training complete in {time.time() - start:.2f} seconds.")

# === Evaluation ===
y_train_pred = rf_model.predict(X_train)
y_train_prob = rf_model.predict_proba(X_train)

y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)

metrics = {
    'train_acc': accuracy_score(y_train, y_train_pred),
    'train_f1': f1_score(y_train, y_train_pred),
    'train_loss': log_loss(y_train, y_train_prob),
    'test_acc': accuracy_score(y_test, y_test_pred),
    'test_f1': f1_score(y_test, y_test_pred),
    'test_loss': log_loss(y_test, y_test_prob),
}
df = pd.DataFrame([metrics])
df.to_csv("rf_metrics_strat_label_patient.csv", index=False)
print("Saved metrics to 'rf_metrics_strat_label_patient.csv'")

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - RF (Strat by Label & Patient)")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_rf_strat_label_patient.svg")


# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - RF (Strat Label & Patient)")
plt.savefig("confusion_matrix_rf_strat_label_patient.svg")
