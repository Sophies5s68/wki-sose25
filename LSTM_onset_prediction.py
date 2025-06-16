import torch 
import torch.nn as nn 
import torch.nn.functional as F

class LSTM_onset(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super.__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = 0.5)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
        