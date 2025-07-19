import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 1D CNN + Transformer für zeitliche EEG-Fenster =====
class CNNTransformer_1D(nn.Module):
    def __init__(self, 
                 num_channels,        # Anzahl der EEG-Kanäle
                 input_length,        # Länge eines einzelnen Fensters in Samples
                 num_classes=2,       # 2 = binäre Klassifikation
                 cnn_hidden_dims=[32, 64],
                 transformer_dim=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 per_window=False):   # True = Onset-Erkennung pro Fenster
        super(CNNTransformer_1D, self).__init__()
        self.per_window = per_window  # Steuert Ausgabeart: pro Fenster oder Sequenz

        # CNN-Block: Extrahiert Merkmale pro Fenster
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, cnn_hidden_dims[0], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_hidden_dims[0], cnn_hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Berechne Ausgabegröße des CNNs zur Projektionsdimension
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, input_length)
            cnn_out = self.cnn(dummy)
            cnn_out_dim = cnn_out.shape[1] * cnn_out.shape[2]
        
        # Lineare Projektion zur Anpassung an Transformer-Eingabe
        self.input_proj = nn.Linear(cnn_out_dim, transformer_dim)

        # Positionskodierung (lernbare Einbettung für Sequenzstruktur)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4000, transformer_dim))  # 4000 = max. Sequenzlänge

        # Transformer-Encoder für zeitliche Abhängigkeiten
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Klassifikationskopf
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, S = x.shape  # B = Batch, T = Fensteranzahl, C = Kanäle, S = Samples
        x = x.view(B * T, C, S)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.input_proj(x)
        x = x.view(B, T, -1)

        # Positionskodierung hinzufügen
        seq_len = x.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb

        # Transformer zur Modellierung von Abhängigkeiten über Zeit
        x = self.transformer_encoder(x)

        if self.per_window:
            return self.classifier(x)  # Ausgabe pro Fenster: [B, T, num_classes]
        else:
            x = x.mean(dim=1)          # Mittelung über alle Fenster
            return self.classifier(x)  # Globale Klassifikation: [B, num_classes]

# ===== 2D CNN + Transformer für Spektrogramme pro Fenster =====
class CNNTransformer_2D(nn.Module):
    def __init__(self, 
                 num_channels,        # Anzahl EEG-Kanäle (C)
                 freq_bins=33,        # Frequenz-Bins im Spektrogramm
                 time_bins=29,        # Zeit-Bins im Spektrogramm
                 num_classes=2,
                 cnn_hidden_dims=[32, 64],
                 transformer_dim=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 per_window=False):   # True = Onset-Klassifikation pro Fenster
        super(CNNTransformer_2D, self).__init__()
        self.per_window = per_window

        # 2D CNN für Verarbeitung von Spektrogrammen: [C, Freq, Time]
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, cnn_hidden_dims[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_hidden_dims[0], cnn_hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Ausgabegröße auf 4x4 normalisieren
        )

        # Berechne CNN-Ausgabegröße für das Transformer-Eingabeformat
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, freq_bins, time_bins)
            dummy_out = self.cnn(dummy)
            cnn_out_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

        # Projektion auf Transformer-Dimension
        self.input_proj = nn.Linear(cnn_out_dim, transformer_dim)

        # Lernbare Positionskodierung
        self.pos_embedding = nn.Parameter(torch.randn(1, 4000, transformer_dim))

        # Transformer zur zeitlichen Modellierung der Fenster
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Klassifikator
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, T, C, Freq, Time] — Sequenz von Spektrogrammen
        """
        B, T, C, F, Tspec = x.shape
        x = x.view(B * T, C, F, Tspec)
        x = self.cnn(x)
        x = x.view(B * T, -1)
        x = self.input_proj(x)
        x = x.view(B, T, -1)

        # Positionale Einbettung hinzufügen
        seq_len = x.size(1)
        pos = self.pos_embedding[:, :seq_len, :]
        x = x + pos

        # Transformer: Modelliert Korrelationen zwischen Fenstern
        x = self.transformer_encoder(x)

        if self.per_window:
            return self.classifier(x)  # Ausgabe pro Fenster
        else:
            x = x.mean(dim=1)          # Durchschnitt über Sequenz
            return self.classifier(x)  # Gesamtklassifikation
