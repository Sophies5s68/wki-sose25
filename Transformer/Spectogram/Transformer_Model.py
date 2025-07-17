import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTransformer_1D(nn.Module):
    def __init__(self, 
                 num_channels,
                 input_length,
                 num_classes=2,
                 cnn_hidden_dims=[32, 64],
                 transformer_dim=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 per_window=False):
        super(CNNTransformer_1D, self).__init__()
        self.per_window = per_window  # set to True for onset prediction
        
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, cnn_hidden_dims[0], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_hidden_dims[0], cnn_hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate CNN output dim
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, input_length)
            cnn_out = self.cnn(dummy)
            cnn_out_dim = cnn_out.shape[1] * cnn_out.shape[2]
        
        self.input_proj = nn.Linear(cnn_out_dim, transformer_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4000, transformer_dim))  # Increased max length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, S = x.shape
        x = x.view(B * T, C, S)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.input_proj(x)
        x = x.view(B, T, -1)

        # Positional embeddings
        seq_len = x.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb

        # Transformer
        x = self.transformer_encoder(x)

        if self.per_window:
            return self.classifier(x)  # [B, T, num_classes]
        else:
            x = x.mean(dim=1)  # [B, transformer_dim]
            return self.classifier(x)  # [B, num_classes]
        

class CNNTransformer_2D(nn.Module):
    def __init__(self, 
                 num_channels,         # EEG channels (C)
                 freq_bins=33,         # Number of frequency bins in spectrogram (F)
                 time_bins=29,         # Number of time bins in spectrogram (T')
                 num_classes=2,
                 cnn_hidden_dims=[32, 64],
                 transformer_dim=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 per_window=False):    # True for onset prediction
        super(CNNTransformer_2D, self).__init__()
        self.per_window = per_window

        # 2D CNN for spectrogram input: [C, F, T']
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, cnn_hidden_dims[0], kernel_size=3, padding=1),  # input: [B*T, C, F, T']
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample both F and T'
            nn.Conv2d(cnn_hidden_dims[0], cnn_hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # output shape: [B*T, 64, 4, 4]
        )

        # Dynamically compute CNN output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, freq_bins, time_bins)
            dummy_out = self.cnn(dummy)
            cnn_out_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

        # Linear projection to Transformer dimension
        self.input_proj = nn.Linear(cnn_out_dim, transformer_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 4000, transformer_dim))  # [1, max_seq_len, D]

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x):
        """
        x shape: [B, T, C, F, T']
        """
        B, T, C, F, Tspec = x.shape
        x = x.view(B * T, C, F, Tspec)     # → [B*T, C, F, T']
        x = self.cnn(x)                    # → [B*T, H, 4, 4]
        x = x.view(B * T, -1)              # → [B*T, cnn_out_dim]
        x = self.input_proj(x)             # → [B*T, transformer_dim]
        x = x.view(B, T, -1)               # → [B, T, transformer_dim]

        # Positional encoding
        seq_len = x.size(1)
        pos = self.pos_embedding[:, :seq_len, :]  # [1, T, D]
        x = x + pos

        # Transformer
        x = self.transformer_encoder(x)  # [B, T, D]

        if self.per_window:
            return self.classifier(x)  # [B, T, num_classes]
        else:
            x = x.mean(dim=1)          # [B, D]
            return self.classifier(x)  # [B, num_classes]