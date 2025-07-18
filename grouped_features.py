import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

## ---- Datei um das Neuronale Netz zu definieren ----
# Modell der Abgabe: CNN_EEG_Conv2d_muster

# Aktivierungsfunktion
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Modell sortiert Features vorher in logische Gruppen, für das bessere Verständnis vom CNN
class CNN_EEG_Conv2d_muster(nn.Module):
    def __init__(self, in_channels=4, n_classes=1):
        super(CNN_EEG_Conv2d_muster, self).__init__()
        
        # Branch 1: 3x3 Faltung
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),  # (B, 32, 6, 5)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # (B, 32, 3, 2)
            nn.Dropout(0.2)
        )
        
        # Branch 2: 5x5 Faltung
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5, 5), padding=2),  # (B, 32, 6, 5)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # (B, 32, 3, 2)
            nn.Dropout(0.2)
        )
        
        # Branch 3: 7x7 Faltung
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), padding=3),  # (B, 32, 6, 5)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # (B, 32, 3, 2)
            nn.Dropout(0.2)
        )
        
        # Kombinierte Ausgabe, weiterverarbeitet durch zusätzliches Conv-Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # (B, 128, 3, 2)
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d((3, 1)),  # (B, 128, 1, 2)
            nn.Dropout(0.25)
        )

        self.flattened_size = 128 * 1 * 2   # Ausgabegröße der letzten Conv-Schicht: (B, 128, 1, 2)
        
        # Klassifikations-MLP 
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
        # Input: (B, 4 Gruppen, 5 Features, 6 Montagen)
        # Umordnen für Conv2D: (B, 4, 6, 5)
        x = x.permute(0, 1, 3, 2)  # (B, 4, 6, 5)
        
        # Drei parallele Branches
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        
        # Channels zusammenführen
        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, 3, 2)
        x = self.conv2(x)                  # (B, 128, 1, 2)
        x = torch.flatten(x, 1)            # (B, 256)
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
    for x, y, *_ in train_loader:
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)  # Labels float und (B,1)

        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            logits = model(x)               # (B, 1)
            probs = torch.sigmoid(logits)
            
            # Prüfung auf numerische Stabilität
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("NaN oder Inf im Modell-Output!")
                print("Min:", probs.min().item(), "Max:", probs.max().item())
                raise ValueError("Ungültige Werte im Modell-Output")

            loss = loss_fn(logits, y)
            loss.backward()
        
        # Verhindert explodierende Gradienten
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = (probs > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(model, test_loader, device='cpu', return_probs=False):
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
    all_probs = []

    with torch.no_grad():
        for x, y, *_ in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.sigmoid(out).view(-1)
            preds = (probs > 0.5).long()
            y = y.long().view(-1)
            correct += (preds == y).sum().item()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            if return_probs:
                all_probs.append(probs.cpu())

    accuracy = correct / len(y_true)
    f1 = f1_score(y_true, y_pred, average='binary')

    if return_probs:
        all_probs = torch.cat(all_probs).numpy()
        return accuracy, y_true, y_pred, f1, all_probs
    else:
        return accuracy, y_true, y_pred, f1