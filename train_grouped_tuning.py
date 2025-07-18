import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from grouped_features import train_model, evaluate_model, CNN_EEG_Conv2d_muster
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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

# ---- Datei zum Trainierne des Modells ----

# ===== Dataset-Klasse =====
# Liest gespeicherte EEG-Samples aus .pt-Dateien, die mehrere Fenster pro Datei enthalten
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
    
     # Gibt die Labels und Gruppen-IDs (eeg_id) aller Samples zurück
    def get_labels_and_groups(self):
        labels = [sample[1] for sample in self.samples]
        groups = [sample[2] for sample in self.samples]
        return labels, groups


# ===== Loss-Funktionen =====
# Dice Loss: robust bei Klassenungleichgewicht, misst Überlappung
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


# Focal Loss: bestraft falsch klassifizierte Beispiele stärker
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

# ===== Early Stopping =====
# Beendet Training frühzeitig, wenn sich der F1-Score nicht verbessert
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

# ===== Threshold-Tuning =====
# Findet optimalen Entscheidungs-Threshold für bestes F1
def threshold_tuning(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device)
            y = y.to(device).float()
            logits = model(x).squeeze()
            probas = torch.sigmoid(logits)
            all_preds.append(probas.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_preds)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    return best_threshold, best_f1, best_precision, best_recall


# ===== Hauptfunktion (Training, Validierung, Tuning, Speichern) =====
def main():
    # Initiale Konfiguration
    data_folder = "data_new_window/win4_step1_groups/" # abgespeicherte extrahierte Features
    run_name = "extra_val_tuning_lr3"
    
    epochs = 50
    batch_size = 1024
    lr = 1e-3
    
    print(f"Training von {run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf Gerät: {device}")
    
    #Lade Datensätze
    file_paths = sorted(glob(os.path.join(data_folder, "*.pt")))
    if not file_paths:
        print(f"[WARNUNG] Keine .pt-Dateien in {data_folder} gefunden – Abbruch.")
        return

    dataset = EEGWindowDatasetCombined(data_folder)
    labels, groups = dataset.get_labels_and_groups()
    torch.backends.cudnn.benchmark = True
    print("Datensatz geladen.")

    # Split 80% Training/Validierung 20% Test
    groups = [sample[2] for sample in dataset] 

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_full, test_idx = next(splitter.split(X=range(len(dataset)), y=labels, groups=groups))

    # Test-Loader vorbereiten
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    # 5 Folds k Fold und trainieren sowie abspeichern von jeweils 5 Modellen
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    threshold_results = []
    all_confusion_matrices = []
    all_best_model_states = []
    all_train_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    all_f1_scores = []

    # Neue Labels + Gruppen für Training-only Set
    train_labels_full = [labels[idx] for idx in train_idx_full]
    train_groups_full = [groups[idx] for idx in train_idx_full]

    # Splitten des 80 Prozentigen Set in Train und Validierung und durchführen eines K-Fold
    for i, (train_idx, val_idx) in enumerate(
            sgkf.split(X=train_idx_full, y=train_labels_full, groups=train_groups_full)):

        train_indices = [train_idx_full[i] for i in train_idx]
        val_indices = [train_idx_full[i] for i in val_idx]

        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True, persistent_workers=True)

        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)

        # Klassenverhältnis berechnen
        train_labels = [labels[idx] for idx in train_indices]
        counter = Counter(train_labels)
        neg, pos = counter[0], counter[1]
        pos_weight = neg / pos if pos > 0 else 1.0
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        
        # Modell, Optimierer, Loss und EarlyStopper
        model = CNN_EEG_Conv2d_muster(in_channels=dataset[0][0].shape[0], n_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = CombinedLoss(pos_weight=pos_weight_tensor, alpha_dice=0.7, alpha_focal=0.3)
        early_stopper = EarlyStopping(patience=5, verbose=True)
        
        # Trainingsloop mit EarlyStopping
        fold_train_losses = []
        fold_train_accuracies = []
        fold_val_accuracies = []
        fold_f1 = []
        
        for epoch in range(epochs):
            train_loss, train_acc = train_model(model, train_loader, optimizer, loss_fn, device)
            val_acc, y_true, y_pred, f1 = evaluate_model(model, val_loader, device)

            print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
                  f"Val Acc={val_acc:.4f} | F1={f1:.4f}")

            fold_train_losses.append(train_loss)
            fold_train_accuracies.append(train_acc)
            fold_val_accuracies.append(val_acc)
            fold_f1.append(f1)
            # Early Stopping wenn F1 5mal nicht verbessert
            early_stopper(f1, model)
            if early_stopper.early_stop:
                print("  --> Early stopping ausgelöst.")
                break

        cm = confusion_matrix(y_true, y_pred)
        all_confusion_matrices.append(cm)
        all_best_model_states.append(early_stopper.best_model_state)
        all_train_losses.append(fold_train_losses)
        all_train_accuracies.append(fold_train_accuracies)
        all_val_accuracies.append(fold_val_accuracies)
        all_f1_scores.append(fold_f1)

        
        torch.cuda.empty_cache()
        gc.collect()
        # Ergebnisse speichern
        model.load_state_dict(early_stopper.best_model_state)
        best_threshold, best_f1, best_prec, best_rec = threshold_tuning(model, test_loader, device)

        print(f"Best Threshold (Fold {i}): {best_threshold:.4f} | F1={best_f1:.4f} | Precision={best_prec:.4f} | Recall={best_rec:.4f}")

        # Speichere Ergebnisse
        threshold_results.append({
            "fold": i,
            "best_threshold": best_threshold,
            "f1": best_f1,
            "precision": best_prec,
            "recall": best_rec
        })

    # Speicherpfade erstellen
    save_path = os.path.join("latest_models", run_name)
    os.makedirs(save_path, exist_ok=True)
    
   
    result_path = os.path.join(save_path, "results")
    os.makedirs(result_path, exist_ok=True)

    # Modelle speichern
    for fold, state_dict in enumerate(all_best_model_states):
        torch.save(state_dict, os.path.join(save_path, f"best_model_fold{fold}.pth"))

    # Trainingsergebnisse als TXT speichern
    report_path = os.path.join(result_path, "training_report.txt")
    with open(report_path, "w") as f:
        for fold in range(len(all_train_losses)):
            f.write(f"\n========== FOLD {fold} ==========\n")
            for epoch in range(len(all_train_losses[fold])):
                f.write(f"Epoch {epoch+1:02d}: "
                        f"Train Loss={all_train_losses[fold][epoch]:.4f} | "
                        f"Train Acc={all_train_accuracies[fold][epoch]:.4f} | "
                        f"Val Acc={all_val_accuracies[fold][epoch]:.4f} | "
                        f"F1={all_f1_scores[fold][epoch]:.4f}\n")

            f.write("\n--- Final Metrics ---\n")
            f.write(f"Final Train Acc: {all_train_accuracies[fold][-1]:.4f}\n")
            f.write(f"Final Val Acc:   {all_val_accuracies[fold][-1]:.4f}\n")
            f.write(f"Final F1 Score:  {all_f1_scores[fold][-1]:.4f}\n")
            f.write("Confusion Matrix letzte Epoche:\n")
            f.write(np.array2string(all_confusion_matrices[fold], separator=', ') + "\n")

    # Trainingsergebnisse als CSV speichern
    csv_path = os.path.join(result_path, "training_metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["config_id", "fold", "epoch", "train_loss", "train_acc", "val_acc", "f1_score"])
        for fold in range(len(all_train_losses)):
            for epoch in range(len(all_train_losses[fold])):
                writer.writerow([
                    run_name,
                    fold,
                    epoch + 1,
                    round(all_train_losses[fold][epoch], 4),
                    round(all_train_accuracies[fold][epoch], 4),
                    round(all_val_accuracies[fold][epoch], 4),
                    round(all_f1_scores[fold][epoch], 4)
                ])

    threshold_csv_path = os.path.join(result_path, "threshold_tuning_results.csv")
    with open(threshold_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["fold", "best_threshold", "f1", "precision", "recall"])
        writer.writeheader()
        for row in threshold_results:
            writer.writerow(row)
            
if __name__ == "__main__":
    main()
