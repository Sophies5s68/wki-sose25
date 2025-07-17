import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CNNTransformerEEG(nn.Module):
    def __init__(
        self,
        in_channels,         # C: number of EEG channels (e.g., 6)
        in_features,         # F: number of features per channel (e.g., 15)
        n_classes=2,         # Output classes: 2 for softmax, 1 for sigmoid
        cnn_out_dim=256,     # Output dimension from CNN
        transformer_dim=256, # Same as CNN output usually
        num_heads=4,
        num_layers=4,
        per_window=False     # True = per-timestep output (onset), False = full-sequence (detection)
    ):
        super().__init__()
        self.per_window = per_window

        # CNN encoder (applied to each window)
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

        # Compute flattened feature size after CNN
        self.flattened_size = self._get_flattened_size(in_channels, in_features)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.flattened_size,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(self.flattened_size)

        # Output head (shared for both modes)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def _get_flattened_size(self, in_channels, in_features):
        # Dummy forward to determine CNN output size
        with torch.no_grad():
            x = torch.zeros(1, in_channels, in_features)  # (B, C, F)
            x = self.cnn(x)  # -> (B, D, L)
            return x.view(1, -1).size(1)  # Flattened size

    def forward(self, x):  # x: (B, T, C, F)
        B, T, C, F = x.shape
        x = x.view(B * T, C, F)        # (B*T, C, F)
        x = self.cnn(x)                # (B*T, D, L)
        x = x.view(B * T, -1)          # (B*T, F_flat)
        x = x.view(B, T, -1)           # (B, T, F_flat)
        
        x = self.pos_encoder(x)     # (B, T, F_flat)

        x = self.transformer(x)        # (B, T, F_flat)

        if self.per_window:
            out = self.classifier(x)   # (B, T, n_classes)
        else:
            out = self.classifier(x[:, -1])  # (B, n_classes)

        return out


