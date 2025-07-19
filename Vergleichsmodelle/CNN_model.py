import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import f1_score

# ---- Datei um die das CNN-Model zu definieren ----

# CNN Model das genutzt wurde um die Auswertung der verschiedenen Parameter zu machen. 
# Dabei wurde noch keine Gruppierung der Features wie in der finalen Abgabe vorgenommen

class CNN_EEG(nn.Module):
    def __init__(self, in_channels, n_classes, activation_fn = nn.ReLU()):
        super(CNN_EEG, self).__init__()
        
        self.activation = activation_fn
        
        self.conv1 = nn.Conv1d(in_channels, out_channels = 64, kernel_size = 3, padding =1)
        self.bn1 = nn.BatchNorm1d(64) # mal ausprobieren
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv1d(64, out_channels= 128, kernel_size = 3, padding =1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)
        
        flattened_size = self._get_flattened_size(in_channels) # Eingangsgröße muss angepasst werden
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    
    def _get_flattened_size(self, in_channels):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, 15)  # Anzahl spectral features, VERÄNDERN JE NACH INPUT
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
        return x.view(1, -1).shape[1]
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(self.dropout1(x))
        x = F.relu(self.bn2(self.conv2(x)))
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
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        probs = torch.sigmoid(logits)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("NaN oder Inf im Output des Modells!")
            print("Minimum:", probs.min().item(), "Maximum:", probs.max().item())
            raise ValueError("Ungültige Werte im Modell-Output")

        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss



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

'''
# alte 
def train_modelold(model, train_loader, optimizer, loss_fn, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # Debug: Check for NaNs or Infs in inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found NaN or Inf in input batch")
            continue
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Found NaN or Inf in label batch")
            continue

        optimizer.zero_grad()
        out = model(x)

        # Debug: Check for NaNs in output
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("Found NaN or Inf in model output")
            continue

        loss = loss_fn(out, y)

        # Debug: Check for NaNs in loss
        if torch.isnan(loss):
            print("Loss is NaN! Skipping this batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    if len(train_loader) == 0 or total == 0:
        return float('nan'), float('nan')

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy
'''