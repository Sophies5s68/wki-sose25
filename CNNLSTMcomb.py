import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, num_channels=21, input_length=512, hidden_dim=128, num_classes=2):
        super(CNN_LSTM, self).__init__()

        # CNN block applied per time step
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # reduce time
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Compute feature size after CNN
        dummy_input = torch.zeros(1, num_channels, input_length)
        cnn_output_dim = self.cnn(dummy_input).view(1, -1).shape[-1]

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, time_steps, channels, samples]
        B, T, C, S = x.shape
        x = x.view(B * T, C, S)            # merge batch and time
        x = self.cnn(x)                    # [B*T, channels_out, reduced_time]
        x = x.view(B, T, -1)               # [batch, time, features]
        lstm_out, _ = self.lstm(x)         # [batch, time, hidden*2]
        out = self.classifier(lstm_out[:, -1])  # last time step
        return out