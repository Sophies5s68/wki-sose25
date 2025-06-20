import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from CNN_model import CNN_EEG, train_model, evaluate_model  # Stelle sicher, dass du die Klasse separat speicherst
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.utils.class_weight import compute_class_weight

class EEGWindowDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        for file in os.listdir(folder_path):
            if file.endswith(".pt"):
                path = os.path.join(folder_path, file)
                features, label, eeg_id, _ = torch.load(path)
                self.samples.append((features, label, eeg_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label, eeg_id= self.samples[idx]  
        
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if features.dim() == 2:
            features = features.unsqueeze(0)  # channel dimension hinzufügen
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def get_labels_and_groups(self):
        labels = [label for _, label, _ in self.samples]
        groups = [eeg_id for _, _, eeg_id in self.samples]
        return labels, groups


def main():

    ordner = "/home/jupyter-wki_team_3/wki-sose25/montage_datasets/"
    unterordner = [f for f in os.listdir(ordner) if os.path.isdir(os.path.join(ordner, f)) and not f.startswith('.')]
    
    epochs = 30
    batch_size = 128
    lr = 1e-3
    
    for config in unterordner:
        print(f"training auf {config}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_train_losses = []
        all_train_accuracies = []
        all_val_accuracies = []
        all_f1_scores = []
        all_confusion_matrices = []
        all_models = []

        data_folder = "/home/jupyter-wki_team_3/wki-sose25/montage_datasets/" + config
        file_paths = sorted(glob(os.path.join(data_folder, "*.pt")))

        if not file_paths:
            print(f"[WARNUNG] Keine .pt-Dateien in {data_folder} gefunden – Ordner wird übersprungen.")

        if not os.path.exists(data_folder):
            raise FileNotFoundError("Unterordner nicht gefunden")
            
        dataset = EEGWindowDataset(data_folder)
        labels, groups = dataset.get_labels_and_groups()
        
        torch.backends.cudnn.benchmark = True
        
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_idx, val_idx) in enumerate(sgkf.split(X=range(len(dataset)), y=labels, groups=groups)):

                 # Metrics tracking
                fold_train_losses = []
                fold_train_accuracies = []
                fold_test_accuracies = []
                fold_f1 = []
                print(f"Training Fold {i}...")
                train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True)
                val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers = 8,pin_memory=True)
                
                
                # Berechnung der Klassengewichte
                train_labels = [labels[idx] for idx in train_idx]
                classes = np.unique(train_labels)
                weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
                class_weights = torch.tensor(weights, dtype=torch.float).to(device)
                
                
                in_channels = dataset[0][0].shape[0]
                model = CNN_EEG(in_channels=in_channels, n_classes=2).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)

                for epoch in range(epochs):
                    train_loss, train_acc = train_model(model, train_loader, optimizer, loss_fn, device)
                    val_acc, y_true, y_pred, f1 = evaluate_model(model, val_loader, device)
                    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | F1={f1:.4f}")
                    fold_train_losses.append(train_loss)
                    fold_train_accuracies.append(train_acc)
                    fold_test_accuracies.append(val_acc)
                    fold_f1.append(f1)

            # Speichern der Metriken
                cm = confusion_matrix(y_true, y_pred)
                all_confusion_matrices.append(cm)
                all_models.append(model)
                all_train_losses.append(fold_train_losses)
                all_train_accuracies.append(fold_train_accuracies)
                all_val_accuracies.append(fold_test_accuracies)
                all_f1_scores.append(fold_f1)

        path = "models_montage/"
        save_path = path + config 
        os.makedirs(save_path, exist_ok=True)
        
        result_path = os.path.join(save_path, "results")
        os.makedirs(result_path, exist_ok=True)

        #Modelle speichern
        for fold, model in enumerate(all_models):
            torch.save(model, os.path.join(save_path, f"model_{fold}.pth"))
        
        report_path = os.path.join(result_path, "training_report.txt")

        with open(report_path, "w") as f:
            for fold in range(len(all_train_losses)):
                f.write(f"\n========== FOLD {fold} ==========\n")
                for epoch in range(epochs):
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

            # Plot metrics per fold
        epoche = list(range(1, epochs + 1))
        for fold in range(len(all_train_losses)):
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            axs[0].plot(epoche, all_train_losses[fold], label='Train Loss')
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title(f"Training Loss - Fold {fold+1}")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(epoche, all_train_accuracies[fold], label='Train Accuracy')
            axs[1].plot(epoche, all_val_accuracies[fold], label='Test Accuracy')
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")
            axs[1].set_title(f"Accuracy - Fold {fold+1}")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"metrics_fold{fold+1}.png"))
            plt.close()

        # Plot average across folds
        mean_train_loss = np.mean(all_train_losses, axis=0)
        mean_train_acc = np.mean(all_train_accuracies, axis=0)
        mean_test_acc = np.mean(all_val_accuracies, axis=0)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].plot(epoche, mean_train_loss, label='Avg Train Loss')
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Average Training Loss Across Folds")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(epoche, mean_train_acc, label='Avg Train Accuracy')
        axs[1].plot(epoche, mean_test_acc, label='Avg Test Accuracy')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_title("Average Training Accuracy Across Folds")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(result_path, "average_metrics.png"))
        plt.close()

            

if __name__ == "__main__":
    main()
