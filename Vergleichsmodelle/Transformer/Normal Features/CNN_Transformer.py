import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ======= Positionskodierung für Transformer =======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Form: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Wird nicht als Parameter trainiert, bleibt aber auf dem Device

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Addiert Positionsembedding auf Zeitachse
        return x


# ======= Modell: CNN + Transformer für EEG =======
# CNN zum extrahieren der Features und anschließendes Klassifizieren im Transformer
class CNNTransformerEEG(nn.Module):
    def __init__(
        self,
        in_channels,         # Anzahl der EEG-Kanäle
        in_features,         # Anzahl der Merkmale pro Kanal
        n_classes=2,         # 2 (Softmax) oder 1 (Sigmoid)
        cnn_out_dim=256,     # Ausgabegröße des CNN (Featuremaps)
        transformer_dim=256, # Eingabegröße des Transformers (muss zum Flattened CNN passen)
        num_heads=4,         # Anzahl der Heads im Transformer
        num_layers=4,        # Anzahl Transformer-Layer
        per_window=False     # True: Klassifikation pro Fenster, False: auf Gesamtsequenz
    ):
        super().__init__()
        self.per_window = per_window

        # CNN-Encoder, verarbeitet jedes Zeitfenster separat
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv1d(64, cnn_out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # Ermitteln der flachen CNN-Ausgabegröße
        self.flattened_size = self._get_flattened_size(in_channels, in_features)

        # Transformer-Encoder mit Positionskodierung
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.flattened_size,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(self.flattened_size)

        # Klassifikationskopf
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def _get_flattened_size(self, in_channels, in_features):
        # Führt eine Dummy-Weiterleitung durch, um die Ausgabegröße des CNNs zu bestimmen
        with torch.no_grad():
            x = torch.zeros(1, in_channels, in_features)
            x = self.cnn(x)              # Form: (1, D, L)
            return x.view(1, -1).size(1) # Flach: (1, D*L)

    def forward(self, x):  # Eingabe x: (B, T, C, F)
        B, T, C, F = x.shape

        # Fensterweise CNN-Verarbeitung
        x = x.view(B * T, C, F)        # (B*T, C, F)
        x = self.cnn(x)                # (B*T, D, L)
        x = x.view(B * T, -1)          # (B*T, D*L)
        x = x.view(B, T, -1)           # (B, T, D*L)

        # Transformer mit Positionskodierung
        x = self.pos_encoder(x)        # (B, T, F)
        x = self.transformer(x)        # (B, T, F)

        # Klassifikation
        if self.per_window:
            out = self.classifier(x)   # (B, T, 1)
        else:
            out = self.classifier(x[:, -1])  # Nur letztes Zeitfenster: (B, 1)

        return out
