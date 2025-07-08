import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, num_channels, input_length, hidden_dim, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # placeholder, will be updated dynamically
        self.lstm = None
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, T, C, S = x.shape
        x = x.view(B * T, C, S)
        x = self.cnn(x)
        x = x.view(B, T, -1)

        # Initialize LSTM if not yet initialized or if input size changed
        if (self.lstm is None) or (self.lstm.input_size != x.shape[2]):
            self.lstm = nn.LSTM(
                input_size=x.shape[2],
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ).to(x.device)

        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out[:, -1, :])
        return out

class CNN_LSTM_2D(nn.Module):
    def __init__(self, num_channels, freq_bins, time_bins, hidden_dim, num_classes=2):
        super(CNN_LSTM, self).__init__()

        # CNN: input shape per window = [channels, freq_bins, time_bins]
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1),  # preserve spatial size
            nn.ReLU(),
            nn.MaxPool2d(2),  # halve freq/time dims
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # halve freq/time dims again
        )

        # Calculate CNN output feature size dynamically
        dummy_input = torch.zeros(1, num_channels, freq_bins, time_bins)
        cnn_out = self.cnn(dummy_input)
        cnn_output_dim = cnn_out.shape[1] * cnn_out.shape[2] * cnn_out.shape[3]  # channels * freq * time

        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=1,
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
        """
        x: [B, T, C, F, Time]  (Batch, Time steps, Channels, Freq bins, Time bins)
        """
        B, T, C, F, Ti = x.shape
        x = x.view(B * T, C, F, Ti)  # merge batch and time steps for CNN
        x = self.cnn(x)              # [B*T, channels_out, F', Ti']

        x = x.view(B, T, -1)         # flatten spatial dims for LSTM input: [B, T, features]

        lstm_out, _ = self.lstm(x)   # [B, T, hidden*2]

        out = self.classifier(lstm_out[:, -1, :])  # use last time step output
        return out