import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)

# === Load Data ===
folder_path = "montage_datasets/combined/win4_step1"
X_all, y_all, pids = [], [], []

for fname in os.listdir(folder_path):
    if fname.endswith(".pt"):
        file_path = os.path.join(folder_path, fname)
        data = torch.load(file_path)
        for sample in data:
            X_all.append(sample[0].flatten())
            y_all.append(sample[1])
            pids.append(sample[2].split("_")[0])  # Patient ID

X_all = np.array(X_all)
y_all = np.array(y_all)
pids = np.array(pids)

print("✅ Loaded all .pt files.")
print("X shape:", X_all.shape)
print("y shape:", y_all.shape)
print("Label distribution:", np.bincount(y_all))


def run_knn_experiment(X_train, X_test, y_train, y_test, tag):
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)

    print(f"\n🚀 Training KNN ({tag})...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"✅ Training complete in {time.time() - start:.2f} seconds.")

    # Evaluation
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    # Metrics
    metrics = {
        'train_acc': accuracy_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_loss': log_loss(y_train, y_train_prob),
        'test_acc': accuracy_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_loss': log_loss(y_test, y_test_prob),
    }
    df = pd.DataFrame([metrics])
    df.to_csv(f"knn_metrics_{tag}.csv", index=False)
    print(f"📄 Saved metrics to knn_metrics_{tag}.csv")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - KNN ({tag})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"roc_curve_knn_{tag}.svg")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - KNN ({tag})")
    plt.savefig(f"confusion_matrix_knn_{tag}.svg")
    plt.close()


# === 1. Stratified by label only ===
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)
run_knn_experiment(X_train, X_test, y_train, y_test, tag="stratlabel")

# === 2. Stratified by patient ===
pid_to_samples = defaultdict(list)
for x, y, pid in zip(X_all, y_all, pids):
    pid_to_samples[pid].append((x, y))

unique_pids = list(pid_to_samples.keys())
pid_major_labels = [
    np.bincount([lbl for _, lbl in pid_to_samples[pid]]).argmax()
    for pid in unique_pids
]

train_pids, test_pids = train_test_split(
    unique_pids, test_size=0.2, stratify=pid_major_labels, random_state=42
)

X_train, y_train, X_test, y_test = [], [], [], []
for pid in train_pids:
    for x, y in pid_to_samples[pid]:
        X_train.append(x)
        y_train.append(y)
for pid in test_pids:
    for x, y in pid_to_samples[pid]:
        X_test.append(x)
        y_test.append(y)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)
run_knn_experiment(X_train, X_test, y_train, y_test, tag="patient_split")


# === 3. Stratified by label and patient ===
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
split = next(sgkf.split(X_all, y_all, groups=pids))
train_idx, test_idx = split
X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]
run_knn_experiment(X_train, X_test, y_train, y_test, tag="strat_label_patient")