import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import f1_score

## ---- Datei um flacheres CNN zu definieren ----


# Flacheres Model mit tanh als Aktivierungsfunktion        
class CNN_EEG_flat(nn.Module):
    def __init__(self, in_channels, n_classes, num_features):
        super(CNN_EEG_flat, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels = 64, kernel_size = 3, padding =1)
        self.bn1 = nn.BatchNorm1d(64) # mal ausprobieren
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv1d(64, out_channels= 128, kernel_size = 3, padding =1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.5)
        
        flattened_size = self._get_flattened_size(in_channels,num_features) # Eingangsgröße muss angepasst werden
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def _get_flattened_size(self, in_channels,num_features):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, num_features)  # Anzahl der spectral features
            x = self.pool1(torch.tanh(self.conv1(x)))
            x = self.pool2(torch.tanh(self.conv2(x)))
        return x.view(1, -1).shape[1]
        
    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = self.pool1(self.dropout1(x))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.pool2(self.dropout2(x))
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

            loss = loss_fn(probs, y)  # Dice Loss erwartet Wahrscheinlichkeiten und Float-Labels
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


