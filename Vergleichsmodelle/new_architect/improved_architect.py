import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

## ---- Datei um das Neuronale Netz zu definieren ----
# Version von CNN_model.py mit variablen Filtergrößen
# hat kleine Verbesserung gebracht, jedoch noch kein 2d Conv

# Hilfsfunktion
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CNN_EEG_Improved(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(CNN_EEG_Improved, self).__init__()
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.LayerNorm([32, 15]),
            nn.SiLU(),  # Swish
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.LayerNorm([32, 15]),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.LayerNorm([32, 15]),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Zweite Convolutional-Stufe nach Zusammenführung
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.LayerNorm([128, 7]),  # 15 → 7 nach MaxPool1d(2)
            nn.SiLU(),
            nn.MaxPool1d(2),  # 7 → 3
            nn.Dropout(0.25)
        )
        
        # Flattened Größe berechnen
        self.flattened_size = 128 * 3  # 128 Kanäle * 3 Zeitschritte

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            Swish(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, in_channels, 15)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, 7)
        x = self.conv2(x)                  # (B, 128, 3)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, optimizer, loss_fn, device='cpu'):
     '''
    Führt einen Trainingsdurchlauf für ein gegebenes Modell auf dem übergebenen DataLoader aus.

    Parameter:
    - model: Das zu trainierende Modell (PyTorch-Modell).
    - train_loader: DataLoader mit Trainingsdaten (x, y).
    - optimizer: PyTorch-Optimizer zur Modellaktualisierung.
    - loss_fn: Verlustfunktion, z. B. BCEWithLogitsLoss.
    - device: Zielgerät ("cpu" oder "cuda").

    Rückgabe:
    - avg_loss: Durchschnittlicher Verlust über alle Batches.
    - accuracy: Klassifikationsgenauigkeit im Training.
    '''
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)  # Labels als Float und [Batch,1]

        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            logits = model(x)               # Roh-Output, Shape [B,1]
            probs = torch.sigmoid(logits)  # Wahrscheinlichkeiten

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("NaN oder Inf im Output des Modells!")
                print("Minimum:", probs.min().item(), "Maximum:", probs.max().item())
                raise ValueError("Ungültige Werte im Modell-Output")

            loss = loss_fn(logits, y)  # Dice Loss erwartet Wahrscheinlichkeiten und Float-Labels
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = (probs > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(model, test_loader, device='cpu'):
    '''
    Bewertet ein trainiertes Modell auf einem Test-/Validierungs-DataLoader.

    Parameter:
    - model: Das zu evaluierende Modell.
    - test_loader: DataLoader mit Testdaten (x, y).
    - device: Zielgerät ("cpu" oder "cuda").
    - return_probs: Wenn True, werden auch die vorhergesagten Wahrscheinlichkeiten zurückgegeben.

    Rückgabe:
    - accuracy: Anteil korrekt klassifizierter Beispiele.
    - y_true: Liste der echten Labels.
    - y_pred: Liste der binären Modellvorhersagen.
    - f1: F1-Score der Vorhersagen.
    - all_probs (optional): Wahrscheinlichkeiten, falls `return_probs=True`.
    '''
    model.eval()
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long().view(-1)  # flatten preds
            y = y.long().view(-1)                   # flatten y
            correct += (preds == y).sum().item()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    accuracy = correct / len(y_true)
    f1 = f1_score(y_true, y_pred, average='binary')
    return accuracy, y_true, y_pred, f1