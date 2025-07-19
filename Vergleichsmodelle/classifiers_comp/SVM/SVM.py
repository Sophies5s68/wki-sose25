import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold



# Load data
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

print("✅ Loaded all .pt files.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label distribution:", np.bincount(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

print("Class weights:", class_weight_dict)
# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SGDClassifier setup
model = SGDClassifier(loss='log_loss', class_weight= class_weight_dict , learning_rate='optimal', random_state=42)

# Training loop
n_epochs = 50
print("Starting epoch-based training...")
classes = np.unique(y_train)

metrics = {
    'epoch': [],
    'train_acc': [],
    'train_f1': [],
    'train_loss': [],
    'test_acc': [],
    'test_f1': [],
    'test_loss': [],
}

for epoch in range(1, n_epochs + 1):
    start = time.time()
    model.partial_fit(X_train, y_train, classes=classes)  # required only for the first call

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_loss = log_loss(y_train, y_train_prob)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_loss = log_loss(y_test, y_test_prob)
    
    metrics['epoch'].append(epoch)
    metrics['train_acc'].append(train_acc)
    metrics['train_f1'].append(train_f1)
    metrics['train_loss'].append(train_loss)
    metrics['test_acc'].append(test_acc)
    metrics['test_f1'].append(test_f1)
    metrics['test_loss'].append(test_loss)

    print(f"\n Epoch {epoch} | ⏱️ Time: {time.time() - start:.2f}s")
    print(f"Train  -> Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Loss: {train_loss:.4f}")
    print(f"Test   -> Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Loss: {test_loss:.4f}")
    
# Convert to DataFrame
df = pd.DataFrame(metrics)

# Save CSV
df.to_csv("training_metrics_svm.csv", index=False)
print("Saved training metrics to 'training_metrics_svm_stratlabel.csv'")

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_f1'], label='Train F1')
plt.plot(df['epoch'], df['test_f1'], label='Test F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_svm_stratlabel.svg")

# Final ROC Curve (test set)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])  # prob for class 1
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve_svm_stratlabel.svg")

print(f"Saved ROC Curve (AUC = {roc_auc:.4f}) to 'roc_curve_svm_stratlabel.svg'")

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Final Confusion Matrix")
plt.savefig("confusion_matrix_svm_stratlabel.svg")


# Stratified by group 
print("Starting Model with stratified by group")
# === Load Data (Group by patient ID) ===
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

# === Stratified patient split ===
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

print("✅ Patient-wise stratified split complete.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {np.bincount(y_train)}, y_test: {np.bincount(y_test)}")

# === Normalize ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Class Weights ===
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)

# === SGDClassifier Setup ===
model = SGDClassifier(loss='log_loss', class_weight=class_weight_dict,
                      learning_rate='optimal', random_state=42)

# === Training Loop ===
n_epochs = 50
print("🚀 Starting epoch-based training...")

metrics = {
    'epoch': [],
    'train_acc': [],
    'train_f1': [],
    'train_loss': [],
    'test_acc': [],
    'test_f1': [],
    'test_loss': [],
}

for epoch in range(1, n_epochs + 1):
    start = time.time()
    model.partial_fit(X_train, y_train, classes=classes)

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_loss = log_loss(y_train, y_train_prob)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_loss = log_loss(y_test, y_test_prob)

    metrics['epoch'].append(epoch)
    metrics['train_acc'].append(train_acc)
    metrics['train_f1'].append(train_f1)
    metrics['train_loss'].append(train_loss)
    metrics['test_acc'].append(test_acc)
    metrics['test_f1'].append(test_f1)
    metrics['test_loss'].append(test_loss)

    print(f"\nEpoch {epoch} ⏱️ {time.time() - start:.2f}s")
    print(f"Train -> Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Loss: {train_loss:.4f}")
    print(f"Test  -> Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Loss: {test_loss:.4f}")

# === Save Metrics ===
df = pd.DataFrame(metrics)
df.to_csv("training_metrics_svm_patient_split.csv", index=False)
print("📄 Saved metrics to 'training_metrics_svm_patient_split.csv'")

# === Plot Accuracy & F1 ===
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_f1'], label='Train F1')
plt.plot(df['epoch'], df['test_f1'], label='Test F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_patient_split.svg")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Patient-Split")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_patient_split.svg")

# === Plot Confusion Matrix ===
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Patient Split")
plt.savefig("confusion_matrix_patient_split.svg")


# Stratified by group and label
print("Start using Label and Patient Stratification")
# === Load Data (Group by patient ID) ===
folder_path = "montage_datasets/combined/win4_step1"
X_all, y_all, pids = [], [], []

for fname in os.listdir(folder_path):
    if fname.endswith(".pt"):
        file_path = os.path.join(folder_path, fname)
        data = torch.load(file_path)
        for sample in data:
            X_all.append(sample[0].flatten())
            y_all.append(sample[1])
            pids.append(sample[2].split("_")[0])  # patient ID before _

X_all = np.array(X_all)
y_all = np.array(y_all)
pids = np.array(pids)

print("✅ Loaded all .pt files.")
print("X shape:", X_all.shape)
print("y shape:", y_all.shape)
print("Label distribution:", np.bincount(y_all))

# === Stratified Split by Label and Patient ===
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X_all, y_all, groups=pids))

X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

print("✅ Stratified by label and grouped by patient.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {np.bincount(y_train)}, y_test: {np.bincount(y_test)}")

# === Normalize ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Class Weights ===
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)

# === SGDClassifier Setup ===
model = SGDClassifier(loss='log_loss', class_weight=class_weight_dict,
                      learning_rate='optimal', random_state=42)

# === Training Loop ===
n_epochs = 50
print("🚀 Starting epoch-based training...")

metrics = {
    'epoch': [],
    'train_acc': [],
    'train_f1': [],
    'train_loss': [],
    'test_acc': [],
    'test_f1': [],
    'test_loss': [],
}

for epoch in range(1, n_epochs + 1):
    start = time.time()
    model.partial_fit(X_train, y_train, classes=classes)

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_loss = log_loss(y_train, y_train_prob)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_loss = log_loss(y_test, y_test_prob)

    metrics['epoch'].append(epoch)
    metrics['train_acc'].append(train_acc)
    metrics['train_f1'].append(train_f1)
    metrics['train_loss'].append(train_loss)
    metrics['test_acc'].append(test_acc)
    metrics['test_f1'].append(test_f1)
    metrics['test_loss'].append(test_loss)

    print(f"\nEpoch {epoch} ⏱️ {time.time() - start:.2f}s")
    print(f"Train -> Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Loss: {train_loss:.4f}")
    print(f"Test  -> Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Loss: {test_loss:.4f}")

# === Save Metrics ===
df = pd.DataFrame(metrics)
df.to_csv("training_metrics_svm_strat_label_patient.csv", index=False)
print("📄 Saved metrics to 'training_metrics_svm_strat_label_patient.csv'")

# === Plot Accuracy & F1 ===
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_f1'], label='Train F1')
plt.plot(df['epoch'], df['test_f1'], label='Test F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_strat_label_patient.svg")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Strat by Label & Patient")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_strat_label_patient.svg")

# === Plot Confusion Matrix ===
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Strat Label & Patient")
plt.savefig("confusion_matrix_strat_label_patient.svg")