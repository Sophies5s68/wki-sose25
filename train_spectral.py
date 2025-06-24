import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from CNN_Model_flat import CNN_EEG_flat, train_model, evaluate_model  # Stelle sicher, dass du die Klasse separat speicherst
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

#kleine Veränderung bei EEGDataset, da jetzt mehrere Daten in einer .pt Datei zusammengefasst, beschleunigt den Ladeprozess

class EEGWindowDatasetCombined(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.samples = []
        for file in os.listdir(folder_path):
            if file.endswith(".pt"):
                samples = torch.load(os.path.join(folder_path, file))
                self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label, eeg_id, *_ = self.samples[idx]
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        #if features.dim() == 2:
            #features = features.unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def get_labels_and_groups(self):
        labels = [label for _, label, *_ in self.samples]
        groups = [eeg_id for _, _, eeg_id, *_ in self.samples]
        return labels, groups



class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # falls preds logits sind
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice
    
# nur dice loss war zu instabil
class CombinedDiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, pos_weight=None):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # BCE Loss
        bce_loss = self.bce(logits, targets.float())

        # Dice Loss
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
    
class FocalLossWithPosWeight(torch.nn.Module):
    def __init__(self, pos_weight=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Sigmoid probs
        probas = torch.sigmoid(preds)

        # Berechne BCE mit pos_weight manuell:
        bce_pos = - self.pos_weight * targets * torch.log(probas + 1e-8)
        bce_neg = - (1 - targets) * torch.log(1 - probas + 1e-8)
        bce = bce_pos + bce_neg

        pt = torch.exp(-bce)  # pt wie in Focal Loss

        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()

class CombinedLoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, alpha_dice=0.5, alpha_focal=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLossWithPosWeight(pos_weight=pos_weight, alpha=focal_alpha, gamma=focal_gamma)
        self.alpha_dice = alpha_dice
        self.alpha_focal = alpha_focal

    def forward(self, preds, targets):
        loss_dice = self.dice_loss(preds, targets)
        loss_focal = self.focal_loss(preds, targets)
        return self.alpha_dice * loss_dice + self.alpha_focal * loss_focal
    
    
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

            
def main():
   
    data_folder = "data_features_sep/spectral/win4_step1_combined" 
    run_name = "spectral_relu"  

    epochs = 50
    batch_size = 128
    lr = 1e-4

    print(f"Training von {run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_paths = sorted(glob(os.path.join(data_folder, "*.pt")))
    if not file_paths:
        print(f"[WARNUNG] Keine .pt-Dateien in {data_folder} gefunden – Abbruch.")
        return

    dataset = EEGWindowDatasetCombined(data_folder)
    labels, groups = dataset.get_labels_and_groups()
    torch.backends.cudnn.benchmark = True
    print("loaded")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    all_confusion_matrices = []
    all_best_model_states = []
    all_train_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    all_f1_scores = []

    for i, (train_idx, val_idx) in enumerate(sgkf.split(X=range(len(dataset)), y=labels, groups=groups)):
        print(f"========== FOLD {i} ==========")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)

        # Pos Weight für Loss
        train_labels = [labels[idx] for idx in train_idx]
        counter = Counter(train_labels)
        neg, pos = counter[0], counter[1]
        pos_weight = neg / pos
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        print(dataset[0][0].shape[0])

        model = CNN_EEG_flat(in_channels=dataset[0][0].shape[0], n_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = CombinedLoss(pos_weight=pos_weight_tensor, alpha_dice=0.7, alpha_focal=0.3) 
        fold_train_losses = []
        fold_train_accuracies = []
        fold_test_accuracies = []
        fold_f1 = []

        early_stopper = EarlyStopping(patience=5, verbose=True)

        for epoch in range(epochs):
            train_loss, train_acc = train_model(model, train_loader, optimizer, loss_fn, device)
            val_acc, y_true, y_pred, f1 = evaluate_model(model, val_loader, device)

            print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
                  f"Val Acc={val_acc:.4f} | F1={f1:.4f}")

            fold_train_losses.append(train_loss)
            fold_train_accuracies.append(train_acc)
            fold_test_accuracies.append(val_acc)
            fold_f1.append(f1)

            early_stopper(f1, model)
            if early_stopper.early_stop:
                print("  --> Early stopping triggered.")
                break

        cm = confusion_matrix(y_true, y_pred)
        all_confusion_matrices.append(cm)
        all_best_model_states.append(early_stopper.best_model_state)
        all_train_losses.append(fold_train_losses)
        all_train_accuracies.append(fold_train_accuracies)
        all_val_accuracies.append(fold_test_accuracies)
        all_f1_scores.append(fold_f1)

    # === SPEICHERN ===
    save_path = os.path.join("models_testing", run_name)
    os.makedirs(save_path, exist_ok=True)

    result_path = os.path.join(save_path, "results")
    os.makedirs(result_path, exist_ok=True)

    for fold, state_dict in enumerate(all_best_model_states):
        torch.save(state_dict, os.path.join(save_path, f"best_model_fold{fold}.pth"))

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
            f.write("Confusion Matrix last epoch:\n")
            f.write(np.array2string(all_confusion_matrices[fold], separator=', ') + "\n")

    # === PLOTs ===
    for fold in range(len(all_train_losses)):
        epochs_fold = list(range(1, len(all_train_losses[fold]) + 1))
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(epochs_fold, all_train_losses[fold], label='Train Loss')
        axs[0].set_title(f"Training Loss - Fold {fold+1}")
        axs[1].plot(epochs_fold, all_train_accuracies[fold], label='Train Acc')
        axs[1].plot(epochs_fold, all_val_accuracies[fold], label='Val Acc')
        axs[1].set_title(f"Accuracy - Fold {fold+1}")
        for ax in axs:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, f"metrics_fold{fold+1}.png"))
        plt.close()

    min_len = min(len(x) for x in all_train_losses)
    mean_train_loss = np.mean([x[:min_len] for x in all_train_losses], axis=0)
    mean_train_acc = np.mean([x[:min_len] for x in all_train_accuracies], axis=0)
    mean_val_acc = np.mean([x[:min_len] for x in all_val_accuracies], axis=0)
    epochs_avg = list(range(1, min_len + 1))

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(epochs_avg, mean_train_loss, label='Avg Train Loss')
    axs[1].plot(epochs_avg, mean_train_acc, label='Avg Train Acc')
    axs[1].plot(epochs_avg, mean_val_acc, label='Avg Val Acc')
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.legend()
        ax.grid(True)
    axs[0].set_title("Average Training Loss")
    axs[1].set_title("Average Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "average_metrics.png"))
    plt.close()
    
    
if __name__ == "__main__":
    main()