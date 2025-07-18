import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from grouped_features import CNN_EEG_Conv2d, train_model, evaluate_model, CNN_EEG_Conv2d_muster
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter
import gc
import csv
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# Dataset zum Laden von gepackten Samples in .pt Dateien (je Datei viele Samples)
class EEGWindowDatasetCombined(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.samples = []  # Liste von Tupeln (x, y, eeg_id, timestamp)
        
        for filename in os.listdir(folder):
            if not filename.endswith(".pt"):
                continue
            data = torch.load(os.path.join(folder, filename))  # dict mit x, y, eeg_id, timestamp
            
            x_all = data["x"]           # Tensor (N, 4, group_features, 6)
            y_all = data["y"]           # Tensor (N,)
            eeg_id_all = data["eeg_id"] # Liste (N)
            timestamp_all = data["timestamp"]  # Liste (N)
            
            for i in range(len(y_all)):
                sample = (
                    x_all[i],           # Tensor (4, group_features, 6)
                    y_all[i].item(),    # Label als int
                    eeg_id_all[i],      # Group (z.B. ID als str oder int)
                    timestamp_all[i],   # Timestamp (z.B. float)
                )
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_labels_and_groups(self):
        labels = [sample[1] for sample in self.samples]
        groups = [sample[2] for sample in self.samples]
        return labels, groups



# Dice Loss für bessere Balance bei Ungleichgewicht
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


# Focal Loss mit pos_weight zur Behandlung von Klassenungleichgewicht
class FocalLossWithPosWeight(nn.Module):
    def __init__(self, pos_weight=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        probas = torch.sigmoid(preds)
        bce_pos = - self.pos_weight * targets * torch.log(probas + 1e-8)
        bce_neg = - (1 - targets) * torch.log(1 - probas + 1e-8)
        bce = bce_pos + bce_neg
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


# Kombinierter Loss aus Dice und Focal Loss
class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=1.0, alpha_dice=0.5, alpha_focal=0.5, focal_alpha=0.5, focal_gamma=2.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLossWithPosWeight(pos_weight=pos_weight, alpha=focal_alpha, gamma=focal_gamma)
        self.alpha_dice = alpha_dice
        self.alpha_focal = alpha_focal

    def forward(self, preds, targets):
        loss_dice = self.dice_loss(preds, targets)
        loss_focal = self.focal_loss(preds, targets)
        return self.alpha_dice * loss_dice + self.alpha_focal * loss_focal


# EarlyStopping basierend auf Validierungs-F1 Score
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_f1, model):
        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} counter (val F1 not improving)")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

def threshold_tuning(probs, labels):
    thresholds = np.linspace(0.05, 0.95, 50)
    best_f1 = 0
    best_threshold = 0.5
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)
        results.append((t, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1, results

def main():
 
    
    data_folder = "data_new_window/win4_step1_groups/"
    run_name = "nofold_final_model_tuning"
    save_dir = os.path.join("models_newWin", run_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EEGWindowDatasetCombined(data_folder)
    labels, groups = dataset.get_labels_and_groups()
    print("loaded")
    labels = np.array(labels)
    groups = np.array(groups)
    unique_groups = np.unique(groups)

    group_to_label = {g: Counter(labels[groups == g]).most_common(1)[0][0] for g in unique_groups}
    group_labels = [group_to_label[g] for g in unique_groups]

    train_groups, val_groups = train_test_split(
        unique_groups, test_size=0.2, stratify=group_labels, random_state=42
    )

    train_idx = [i for i in range(len(dataset)) if groups[i] in train_groups]
    val_idx = [i for i in range(len(dataset)) if groups[i] in val_groups]

    train_labels = [labels[i] for i in train_idx]
    label_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
    weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=int(2 * len(train_labels)), replacement=True)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1024, sampler=sampler, num_workers=8)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1024, shuffle=False, num_workers=8)

    pos_weight = label_counts[0] / label_counts[1] if label_counts[1] > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)

    model = CNN_EEG_Conv2d_muster(in_channels=dataset[0][0].shape[0], n_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    loss_fn = CombinedLoss(pos_weight=pos_weight_tensor, alpha_dice=0.7, alpha_focal=0.3)
    early_stopper = EarlyStopping(patience=5, verbose=True)

    train_losses, train_accuracies, val_accuracies, f1_scores = [], [], [], []
    all_val_probs, all_val_labels = [], []
    print(f"start training on {device}")
    for epoch in range(50):
        train_loss, train_acc = train_model(model, train_loader, optimizer, loss_fn, device)
        val_acc, y_true, y_pred, f1, val_probs = evaluate_model(model, val_loader, device,return_probs = True)

        all_val_probs.append(val_probs)
        all_val_labels.append(torch.tensor(y_true))

        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | F1={f1:.4f}")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        f1_scores.append(f1)

        early_stopper(f1, model)
        if early_stopper.early_stop:
            print("  --> Early stopping ausgelöst.")
            break
    model.load_state_dict(early_stopper.best_model_state)
    model.eval()

    # Vorhersagen für Threshold-Tuning sammeln
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch, *_ in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch).squeeze()
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(y_batch)

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Threshold-Tuning: optimaler F1-Score
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics_list = []

    best_f1 = 0
    best_thresh = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_targets, preds)
        precision = precision_score(all_targets, preds, zero_division=0)
        recall = recall_score(all_targets, preds, zero_division=0)
        metrics_list.append([t, f1, precision, recall])

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_precision = precision
            best_recall = recall

    # Speichern: Zusammenfassung (TXT)
    results_path = os.path.join(save_dir, "threshold_results.txt")
    csv_path = os.path.join(save_dir, "threshold_metrics.csv")
    np.save(os.path.join(save_dir, "val_probs.npy"), all_probs)
    np.save(os.path.join(save_dir, "val_labels.npy"), all_targets)
    with open(results_path, "w") as f:
        f.write(f"Best Threshold: {best_thresh:.4f}\n")
        f.write(f"F1 Score:       {best_f1:.4f}\n")
        f.write(f"Precision:      {best_precision:.4f}\n")
        f.write(f"Recall:         {best_recall:.4f}\n")

    # Speichern: CSV mit allen Threshold-Metriken
    csv_path = os.path.join(save_dir, "threshold_metrics.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Threshold", "F1", "Precision", "Recall"])
        for row in metrics_list:
            writer.writerow([f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}"])

    # Speichern der Wahrscheinlichkeiten und Labels
    np.save(os.path.join(save_dir, "val_probs.npy"), all_probs)
    np.save(os.path.join(save_dir, "val_labels.npy"), all_targets)

if __name__ == "__main__":
    main()
