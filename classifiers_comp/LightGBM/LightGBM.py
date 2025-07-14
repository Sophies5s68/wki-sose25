import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from lightgbm import LGBMClassifier

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

# === Optional: Normalize (not strictly needed for LightGBM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === LightGBM Classifier ===
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # handle imbalance

lgb_model = LGBMClassifier(
    n_estimators=100,
    class_weight=None,  # we use scale_pos_weight instead
    scale_pos_weight=pos_weight,
    objective='binary',
    random_state=42,
    n_jobs=-1
)

print("🚀 Training LightGBM...")
start = time.time()
lgb_model.fit(X_train, y_train)
print(f" Training complete in {time.time() - start:.2f} seconds.")

# === Evaluation ===
y_train_pred = lgb_model.predict(X_train)
y_train_prob = lgb_model.predict_proba(X_train)

y_test_pred = lgb_model.predict(X_test)
y_test_prob = lgb_model.predict_proba(X_test)

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
df.to_csv("lgbm_metrics.csv", index=False)
print(" Saved LightGBM metrics to 'lgbm_metrics.csv'")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LightGBM")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_lgbm.svg")


# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - LightGBM")
plt.savefig("confusion_matrix_lgbm.svg")


from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold

# === Stratified by Patient ID ===
print("Starting LightGBM with stratified patient ID split...")
pid_to_samples = defaultdict(list)

for fname in os.listdir(folder_path):
    if fname.endswith(".pt"):
        file_path = os.path.join(folder_path, fname)
        data = torch.load(file_path)
        for sample in data:
            features = sample[0].flatten()
            label = sample[1]
            patient_id = sample[2].split("_")[0]
            pid_to_samples[patient_id].append((features, label))

# Create split
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

print(f"Patient-split: X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Label dist (train): {np.bincount(y_train)}, (test): {np.bincount(y_test)}")

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Recalculate pos_weight
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train LightGBM
lgb_model = LGBMClassifier(
    n_estimators=100,
    scale_pos_weight=pos_weight,
    objective='binary',
    random_state=42,
    n_jobs=-1
)
print("🚀 Training LightGBM (patient strat)...")
start = time.time()
lgb_model.fit(X_train, y_train)
print(f"✅ Training done in {time.time() - start:.2f} seconds.")

# Evaluation
y_test_pred = lgb_model.predict(X_test)
y_test_prob = lgb_model.predict_proba(X_test)

metrics = {
    'test_acc': accuracy_score(y_test, y_test_pred),
    'test_f1': f1_score(y_test, y_test_pred),
    'test_loss': log_loss(y_test, y_test_prob)
}
pd.DataFrame([metrics]).to_csv("lgbm_metrics_patient_split.csv", index=False)
print("📄 Saved metrics to 'lgbm_metrics_patient_split.csv'")

# ROC
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC - LightGBM (Patient Split)")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_lgbm_patient_split.svg")
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - LightGBM (Patient Split)")
plt.savefig("confusion_matrix_lgbm_patient_split.svg")
plt.show()


# === Stratified by Label and Patient (StratifiedGroupKFold) ===
print("Starting LightGBM with stratified group by label and patient...")

X_all, y_all, pids = [], [], []
for fname in os.listdir(folder_path):
    if fname.endswith(".pt"):
        file_path = os.path.join(folder_path, fname)
        data = torch.load(file_path)
        for sample in data:
            X_all.append(sample[0].flatten())
            y_all.append(sample[1])
            pids.append(sample[2].split("_")[0])

X_all = np.array(X_all)
y_all = np.array(y_all)
pids = np.array(pids)

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X_all, y_all, groups=pids))

X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

print(f"StratGroupSplit: X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Label dist (train): {np.bincount(y_train)}, (test): {np.bincount(y_test)}")

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Recalculate pos_weight
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train LightGBM
lgb_model = LGBMClassifier(
    n_estimators=100,
    scale_pos_weight=pos_weight,
    objective='binary',
    random_state=42,
    n_jobs=-1
)
print("Training LightGBM (strat label + patient)...")
start = time.time()
lgb_model.fit(X_train, y_train)
print(f"Training done in {time.time() - start:.2f} seconds.")

# Evaluation
y_test_pred = lgb_model.predict(X_test)
y_test_prob = lgb_model.predict_proba(X_test)

metrics = {
    'test_acc': accuracy_score(y_test, y_test_pred),
    'test_f1': f1_score(y_test, y_test_pred),
    'test_loss': log_loss(y_test, y_test_prob)
}
pd.DataFrame([metrics]).to_csv("lgbm_metrics_strat_label_patient.csv", index=False)
print("Saved metrics to 'lgbm_metrics_strat_label_patient.csv'")

# ROC
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC - LightGBM (Strat Label + Patient)")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_lgbm_strat_label_patient.svg")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - LightGBM (Strat Label + Patient)")
plt.savefig("confusion_matrix_lgbm_strat_label_patient.svg")
