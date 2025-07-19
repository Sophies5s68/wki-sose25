import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

''' --- Alternative Modelle, die ausprobiert wurden, um Performance von verschiedenen Modellen und Paramtern zu vergleichen ---'''
# Flacht Input- Matrix (channels, features) pro Gruppe auf und führt 1D-Faltung aus
class CNN_EEG_Improved(nn.Module):
    def __init__(self, in_channels=4, n_classes=1):
        super(CNN_EEG_Improved, self).__init__()
        
        # Erwarteter Input: (Batch, 4 Gruppen, 5 Features, 6 Kanäle)
        # wird flach gemacht zu (Batch, 4, 30) = 30 Merkmale pro Gruppe
        
        # Branch 1: 3x3 Faltung
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),  # (B, 32, 5)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # (B, 32, 2)
            nn.Dropout(0.2)
        )
        
        # Branch 2: 5x5 Faltung
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),  # (B, 32, 5)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # (B, 32, 2)
            nn.Dropout(0.2)
        )
        # Branch 3: 7x7 Faltung
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),  # (B, 32, 5)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # (B, 32, 2)
            nn.Dropout(0.2)
        )
        
        # Nach Concatenation: Channels = 32 * 3 = 96, Länge = 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),  # (B, 128, 2)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(), 
            nn.MaxPool1d(2),  # (B, 128, 1) - Achtung: 2 // 2 = 1
            nn.Dropout(0.25)
        )
        
        self.flattened_size = 128 * 7  # Channels * Länge = 128
        
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
        # Input: (B, 4, 5, 6)
        # Erst flach machen: Features × Kanäle = 30 Werte pro Gruppe (B, 4, 30)
        x = x.view(x.size(0), 4, -1)  # (B, 4 Kanäle (Gruppen), 30 Werte)

        # Drei parallele Convolutional-Pfade
        x3 = self.branch3(x)  
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, ...)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Modell ähnelt dem verwendeten Modell, benutzt aber die originale Struktur der Daten und sortiert sie nicht vorher in logische Gruppen
class CNN_EEG_Conv2d(nn.Module):
    def __init__(self, in_channels=4, n_classes=1):
        super(CNN_EEG_Conv2d, self).__init__()
        
        # Branch 1: 3x3 Faltung
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), padding=(1,1)),  # (B, 32, 5, 6)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2,2)),  # (B, 32, 2, 3)
            nn.Dropout(0.2)
        )
        
        # Branch 2: 5x5 Faltung
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5,5), padding=(2,2)),  # (B, 32, 5, 6)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2,2)),  # (B, 32, 2, 3)
            nn.Dropout(0.2)
        )
        
        # Branch 3: 7x7 Faltung
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
        # Input: (B, 4, 5, 6)
        
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        
        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, 2, 3)
        x = self.conv2(x)                   # (B, 128, 1, 3)
        x = torch.flatten(x, 1)             # (B, 384)
        x = self.classifier(x)
        return x
