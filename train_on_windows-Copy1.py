    #stratified group split

import random
import importlib
import torch
import os
from torch.utils.data import random_split, DataLoader, ConcatDataset, Subset,TensorDataset
import matplotlib.pyplot as plt
from collections import Counter
from glob import glob
import torch.nn as nn 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score
import csv
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# Datenordner einladen:
data_folder = "data_test"
file_paths = sorted(glob(os.path.join(data_folder, "*.pt")))

if not os.path.exists(data_folder):
    raise FileNotFoundError("Unterordner nicht gefunden")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Modell instantiieren

train_losses = []
train_accuracies = []
test_accuracies = []
metrics = []
batch_nr = 0
train_dataset_global = []
test_dataset_global =[]

all_x = []
all_y = []
all_id = []

for file_path in file_paths:
    dataset = torch.load(file_path)
    for x, y, gruppe in dataset:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        all_x.append(x)
        all_y.append(int(y))
        all_id.append(gruppe)
        batch_nr = batch_nr + 1

# In NumPy konvertieren
all_x_np = np.stack([x.numpy() for x in all_x])
all_y_np = np.array(all_y)
all_id_np = np.array(all_id)
    
# DataFrame erstellen
df = pd.DataFrame({
    'x': list(all_x_np),  # wichtig: Liste von Arrays
    'y': all_y_np,
    'id': all_id_np
})
# stratified == erhält Klassengewichtung für alle Folds und Groupkfold = keine Überschneidung Patienten
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

num_epochs = 30
for fold, (train_idx, test_idx) in enumerate(cv.split(df['x'], df['y'], df['id'])):
    print(f"\n=== Fold {fold+1} ===")

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    '''
    # Balancieren der Testdaten
    train_pos = train_df[train_df['y'] == 1]
    train_neg = train_df[train_df['y'] == 0].sample(len(train_pos), random_state=42)
    train_bal = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)
    
    X_train = np.stack(train_bal['x'].values)
    y_train = train_bal['y'].values
    
    
    X_test = np.stack(test_df['x'].values)
    y_test = test_df['y'].values
    '''
    X_train = np.stack(train_df['x'].values)
    y_train = train_df['y'].values
    
    X_test = np.stack(test_df['x'].values)
    y_test = test_df['y'].values
    
    # Berechnung der Klassengewichte
    classes = np.unique(all_y_np)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    for cls, w in zip(classes, weights):
        print(f"Class {cls}: weight = {w:.4f}")

    #Modell instantiieren
    import CNN_model
    importlib.reload(CNN_model)
    model = CNN_model.CNN_EEG(in_channels=18, n_classes=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-4)
    num_epochs = 30


    # Wenn X_train und y_train numpy arrays sind:
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Gleiches für Testdaten:
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for x, y in train_loader:
        print("x NaN:", torch.isnan(x).any())
        print("x Inf:", torch.isinf(x).any())
        print("y NaN:", torch.isnan(y).any())
        print("y Inf:", torch.isinf(y).any())
        print("x stats - min:", x.min().item(), "max:", x.max().item(), "mean:", x.mean().item(), "std:", x.std().item())
        break

    print(f"starting training on {device}")
    
    # Metrics tracking
    fold_train_losses = []
    fold_train_accuracies = []
    fold_test_accuracies = []
    fold_test_f1 = []

    #Training 
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = CNN_model.train_model(model, train_loader, optimizer, loss_fn,device)
        test_acc, y_true, y_pred, f1_score = CNN_model.evaluate_model(model, test_loader,device)
        
        fold_train_losses.append(train_loss)
        fold_train_accuracies.append(train_acc)
        fold_test_accuracies.append(test_acc)
        fold_test_f1.append(f1_score)
        print(f"Metrics of epoch {epoch}:,fold: {fold} test_acc: {test_acc}, train_acc: {train_acc}, f1-score: {f1_score}")
    # Save metrics for this fold
    train_losses.append(fold_train_losses)
    train_accuracies.append(fold_train_accuracies)
    test_accuracies.append(fold_test_accuracies)
     
    # Confusion Matrix of one fold
    cm = confusion_matrix(y_true, y_pred)
    metrics.append((test_acc,train_acc,y_pred,cm))

    print(f"Metrics last epoch,fold: {fold} test_acc: {test_acc}, train_acc: {train_acc}")
    
    
    data = data_folder
    path = "models_strat/"
    save_path = path + data #Hier ändern für Ordner
    os.makedirs(save_path, exist_ok=True)  # Verzeichnis erstellen, falls es noch nicht existiert

    torch.save(model, os.path.join(save_path, f"model_{fold}.pth"))
    
    
    
    print("finished training")
    result_path = os.path.join(save_path, "results")
    os.makedirs(result_path, exist_ok=True)
    report_path = os.path.join(result_path, "training_report.txt")
    with open(report_path, "w") as f:
        # Print final metrics and confusion matrix
        for fold, (test_acc, train_acc, y_pred, cm) in enumerate(metrics):
            f.write(f"Fold {fold+1}\n")
            f.write(f"  Test accuracy:  {test_acc:.2f}\n")
            f.write(f"  Train accuracy: {train_acc:.2f}\n\n")

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Kein Anfall", "Anfall"])
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
            ax.set_title("Confusion Matrix (Test Set)")
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"confusion_matrix_fold{fold+1}.png"))
            plt.close()

        # Plot metrics per fold
        epochs = list(range(1, num_epochs + 1))
        for fold in range(len(train_losses)):
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            axs[0].plot(epochs, train_losses[fold], label='Train Loss')
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title(f"Training Loss - Fold {fold+1}")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(epochs, train_accuracies[fold], label='Train Accuracy')
            axs[1].plot(epochs, test_accuracies[fold], label='Test Accuracy')
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")
            axs[1].set_title(f"Accuracy - Fold {fold+1}")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"metrics_fold{fold+1}.png"))
            plt.close()

        # Plot average across folds
        mean_train_loss = np.mean(train_losses, axis=0)
        mean_train_acc = np.mean(train_accuracies, axis=0)
        mean_test_acc = np.mean(test_accuracies, axis=0)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].plot(epochs, mean_train_loss, label='Avg Train Loss')
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Average Training Loss Across Folds")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(epochs, mean_train_acc, label='Avg Train Accuracy')
        axs[1].plot(epochs, mean_test_acc, label='Avg Test Accuracy')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_title("Average Training Accuracy Across Folds")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(result_path, "average_metrics.png"))
        plt.close()