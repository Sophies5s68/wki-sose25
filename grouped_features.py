import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

# Modell der Abgabe

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
'''
# Test einer CNN Architektur mit gruppierten Features und 1D Convolution

class CNN_EEG_Improved(nn.Module):
    def __init__(self, in_channels=4, n_classes=1):
        super(CNN_EEG_Improved, self).__init__()
        
        # Input: (Batch, 4, 5)
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),  # (B, 32, 5)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(), # 2d_grouped_muster war mit SiLU, das hier ist 2d_grouped_muster_bslract 
            nn.MaxPool1d(2),  # (B, 32, 2)
            nn.Dropout(0.2)
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),  # (B, 32, 5)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(), # 2d_grouped_muster war mit SiLU
            nn.MaxPool1d(2),  # (B, 32, 2)
            nn.Dropout(0.2)
        )
        
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),  # (B, 32, 5)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(), # 2d_grouped_muster war mit SiLU
            nn.MaxPool1d(2),  # (B, 32, 2)
            nn.Dropout(0.2)
        )
        
        # Nach Concatenation: Channels = 32 * 3 = 96, Länge = 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),  # (B, 128, 2)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(), # 2d_grouped_muster war mit SiLU
            nn.MaxPool1d(2),  # (B, 128, 1) - Achtung: 2 // 2 = 1
            nn.Dropout(0.25)
        )
        
        self.flattened_size = 128 * 7  # Channels * Länge = 128
        
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
        # x: (B, 4 Gruppen, 5 Features, 6 Kanäle)

        # Erst flach machen: Features × Kanäle = 30 Werte pro Gruppe
        x = x.view(x.size(0), 4, -1)  # (B, 4 Kanäle (Gruppen), 30 Werte)

        # Jetzt kannst du direkt in branch3, branch5, branch7 gehen
        x3 = self.branch3(x)  # Conv1d(4, ...)
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, ...)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
'''

'''
# Architekturansatz mit 2d Convolution

class CNN_EEG_Conv2d(nn.Module):
    def __init__(self, in_channels=4, n_classes=1):
        super(CNN_EEG_Conv2d, self).__init__()
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), padding=(1,1)),  # (B, 32, 5, 6)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2,2)),  # (B, 32, 2, 3)
            nn.Dropout(0.2)
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5,5), padding=(2,2)),  # (B, 32, 5, 6)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2,2)),  # (B, 32, 2, 3)
            nn.Dropout(0.2)
        )
        
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7,7), padding=(3,3)),  # (B, 32, 5, 6)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2,2)),  # (B, 32, 2, 3)
            nn.Dropout(0.2)
        )
        
        # Nach Concatenation: Channels = 32 * 3 = 96, Height = 2, Width = 3
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # (B, 128, 2, 3)
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d((2,1)),  # (B, 128, 1, 3)
            nn.Dropout(0.25)
        )
        
        # Flatten: 128 Channels * 1 Height * 3 Width = 384
        self.flattened_size = 128 * 1 * 3
        
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
        # x: (B, 4, 5, 6)
        
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        
        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, 2, 3)
        x = self.conv2(x)                   # (B, 128, 1, 3)
        x = torch.flatten(x, 1)             # (B, 384)
        x = self.classifier(x)
        return x
'''

# Aktuelles Modell
# nutzt variable Filterlängen und 2d Convolution
# perumtiert den Input um (B, 4, 6, 5) zu erzeugen
# soll 
class CNN_EEG_Conv2d_muster(nn.Module):
    def __init__(self, in_channels=4, n_classes=1):
        super(CNN_EEG_Conv2d_muster, self).__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),  # (B, 32, 6, 5)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # (B, 32, 3, 2)
            nn.Dropout(0.2)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5, 5), padding=2),  # (B, 32, 6, 5)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # (B, 32, 3, 2)
            nn.Dropout(0.2)
        )

        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), padding=3),  # (B, 32, 6, 5)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # (B, 32, 3, 2)
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # (B, 128, 3, 2)
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d((3, 1)),  # (B, 128, 1, 2)
            nn.Dropout(0.25)
        )

        self.flattened_size = 128 * 1 * 2

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
        # Erwartet: x: (B, 4 Gruppen, 5 Features, 6 Montagen)
        # Wir permutieren, damit es (B, 4 Gruppen, 6 Montagen, 5 Features) wird:
        x = x.permute(0, 1, 3, 2)  # (B, 4, 6, 5)

        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, 3, 2)
        x = self.conv2(x)                  # (B, 128, 1, 2)
        x = torch.flatten(x, 1)            # (B, 256)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, optimizer, loss_fn, device='cpu'):
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

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("NaN oder Inf im Modell-Output!")
                print("Min:", probs.min().item(), "Max:", probs.max().item())
                raise ValueError("Ungültige Werte im Modell-Output")

            loss = loss_fn(logits, y)
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


def evaluate_model(model, test_loader, device='cpu', return_probs=False):
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