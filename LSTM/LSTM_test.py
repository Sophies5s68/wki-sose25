import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os


class OnsetLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2):
        super(OnsetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)              # (B, T, H*2)
        out = self.classifier(out).squeeze(-1)  # (B, T)
        return out  # Logits, keine Sigmoid hier!

# Dataset-Klasse zum Laden der gespeicherten .pt-Batches
class EEGOnsetBatchDataset(Dataset):
    def __init__(self, batch_folder):
        self.batch_paths = sorted([os.path.join(batch_folder, f) for f in os.listdir(batch_folder) if f.endswith('.pt')])

    def __len__(self):
        return len(self.batch_paths)

    def __getitem__(self, idx):
        return torch.load(self.batch_paths[idx])

# Training & Evaluation
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        mask = torch.arange(x.shape[1], device=device)[None, :] < batch['lengths'][:, None].to(device)  # (B, T)

        optimizer.zero_grad()
        logits = model(x)  # (B, T)
        loss = criterion(logits[mask], y[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        mask = torch.arange(x.shape[1], device=device)[None, :] < batch['lengths'][:, None].to(device)

        logits = model(x)
        loss = criterion(logits[mask], y[mask])
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Hauptfunktion
def run_training(train_dir, val_dir, epochs=10, batch_size=1, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    train_dataset = EEGOnsetBatchDataset(train_dir)
    val_dataset = EEGOnsetBatchDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size = None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None)

    model = OnsetLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "onset_lstm.pt")
    print("Modell gespeichert.")

    
def main():
    # Beispielaufruf
    run_training("LSTM/tensor_batches/train", "LSTM/tensor_batches/val", epochs=15)




if __name__ == "__main__":
    main()