import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from grouped_features import CNN_EEG_Conv2d, train_model, CNN_EEG_Conv2d_muster, CNN_EEG_Improved
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter
import gc
import csv
from torch.utils.data import WeightedRandomSampler

# Dataset zum Laden von gepackten Samples in .pt Dateien (je Datei viele Samples)
class EEGWindowDatasetCombined(torch.utils.data.Dataset):
    def __init__(self, folder, augment=False):
        self.samples = []
        self.augment = augment

        for filename in os.listdir(folder):
            if not filename.endswith(".pt"):
                continue
            data = torch.load(os.path.join(folder, filename))
            x_all = data["x"]
            y_all = data["y"]
            eeg_id_all = data["eeg_id"]
            timestamp_all = data["timestamp"]

            for i in range(len(y_all)):
                self.samples.append((x_all[i], y_all[i].item(), eeg_id_all[i], timestamp_all[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, eeg_id, ts = self.samples[idx]

        if self.augment and y == 1:
            if torch.rand(1).item() < 0.4:
                x = add_noise(x)
            if torch.rand(1).item() < 0.2:
                x = time_mask(x)
            if torch.rand(1).item() < 0.2:
                x = segment_permutation(x)
            if torch.rand(1).item() < 0.2:
                x = channel_dropout(x)

            # Mixup with another random positive
            if torch.rand(1).item() < 0.2:
                other_idx = torch.randint(0, len(self.samples), (1,)).item()
                x_mix, _ = mixup_positive((x, y, eeg_id, ts), self.samples[other_idx])
                x = x_mix

        return x, y, eeg_id, ts

    def get_labels_and_groups(self):
        labels = [sample[1] for sample in self.samples]
        groups = [sample[2] for sample in self.samples]
        return labels, groups

def add_noise(x, mean=0.0, std=0.01):
    return x + torch.randn_like(x) * std

def time_mask(x, max_width=2):
    x = x.clone()
    T = x.shape[-1]
    width = np.random.randint(1, max_width + 1)
    start = np.random.randint(0, T - width + 1)
    x[..., start:start+width] = 0
    return x

def segment_permutation(x, segments=3):
    x = x.clone()
    T = x.shape[-1]
    if T % segments != 0:
        return x
    seg_size = T // segments
    idx = torch.randperm(segments)
    return torch.cat([x[..., i*seg_size:(i+1)*seg_size] for i in idx], dim=-1)

def channel_dropout(x, p=0.25):
    """ Randomly zero out 1 EEG channel (i.e., one of the 4 groups) """
    if torch.rand(1).item() < p:
        idx = torch.randint(0, x.shape[0], (1,))
        x[idx] = 0
    return x

def mixup_positive(sample1, sample2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    x1, y1, _, _ = sample1
    x2, y2, _, _ = sample2
    if y1 == 1 and y2 == 1:
        x_mix = lam * x1 + (1 - lam) * x2
        return x_mix, 1
    return x1, y1



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
from sklearn.metrics import f1_score

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for x, y, *_ in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.sigmoid(out).view(-1)
            y_true.extend(y.cpu().tolist())
            y_probs.extend(probs.cpu().tolist())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # ✅ Threshold tuning
    best_thresh = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (y_probs > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    final_preds = (y_probs > best_thresh).astype(int)
    accuracy = (final_preds == y_true).mean()

    return accuracy, y_true.tolist(), final_preds.tolist(), best_f1

def main():
    data_folder = "data_new_window/win4_step1_groups/"
    run_name = "2d_grouped_strong_augmentation_no_sampler_treshhold_tuning_muster_loss_angepasst"

    epochs = 50
    batch_size = 1024
    lr = 1e-4

    print(f"Training von {run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf Gerät: {device}")

    file_paths = sorted(glob(os.path.join(data_folder, "*.pt")))
    if not file_paths:
        print(f"[WARNUNG] Keine .pt-Dateien in {data_folder} gefunden – Abbruch.")
        return

    dataset = EEGWindowDatasetCombined(data_folder)
    labels, groups = dataset.get_labels_and_groups()
    torch.backends.cudnn.benchmark = True
    print("Datensatz geladen.")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    all_confusion_matrices = []
    all_best_model_states = []
    all_train_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    all_f1_scores = []

    for i, (train_idx, val_idx) in enumerate(sgkf.split(X=range(len(dataset)), y=labels, groups=groups)):
        print(f"========== FOLD {i} ==========")
    
        # Augmentation for positive samples only
        augmented_dataset = EEGWindowDatasetCombined(data_folder, augment=True)
        train_subset = Subset(augmented_dataset, train_idx)

        # Use standard DataLoader with shuffle
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        train_labels = [labels[idx] for idx in train_idx]
        counter = Counter(train_labels)
        print(f"Fold {i}: Class distribution: {counter}")
        neg, pos = counter[0], counter[1]
        pos_weight = neg / pos if pos > 0 else 1.0
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        
        # Erstmal ausprobieren, ob mit pos weight = 1, da sonst overfitting passiernen könnte
        #pos_weight_tensor = torch.tensor([1.0], dtype=torch.float32).to(device)
        model = CNN_EEG_Conv2d_muster(in_channels=dataset[0][0].shape[0], n_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        loss_fn = CombinedLoss(pos_weight=pos_weight_tensor, alpha_dice=0.8, alpha_focal=0.2, focal_gamma=1.0)

        early_stopper = EarlyStopping(patience=5, verbose=True)

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

        del train_loader, val_loader, model
        torch.cuda.empty_cache()
        gc.collect()

    # Speicherpfade erstellen
    save_path = os.path.join("models_newWin", run_name)
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


if __name__ == "__main__":
    main()
