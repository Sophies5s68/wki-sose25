import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CNN_EEG(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(CNN_EEG, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels = 64, kernel_size = (3,3), padding =1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, out_channels= 128, kernel_size = (3,3), padding =1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_flattened_size(in_channels), 256) # Eingangsgröße muss angepasst werden
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    
    def _get_flattened_size(self, in_channels):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, 5, 5)  # your brain map size
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
        return x.view(1, -1).shape[1]
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))        
        x = self.dropout(x)
        x = self.fc2(x)                
        return x
    

def train_model(model, train_loader, optimizer, loss_fn, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            out = model(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("NaN oder Inf im Output des Modells!")
                print("Minimum:", out.min().item(), "Maximum:", out.max().item())
                raise ValueError("Ungültige Werte im Modell-Output")
            loss = loss_fn(out, y)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    loss = total_loss / len(train_loader)
    accuracy = correct / total
    return loss , accuracy

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            pred = (probs[:, 1] > 0.3).long()
            correct += (pred == y).sum().item()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    accuracy = correct / len(y_true)
    return accuracy, y_true, y_pred

def train_modelold(model, train_loader, optimizer, loss_fn, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # Debug: Check for NaNs or Infs in inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found NaN or Inf in input batch")
            continue
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Found NaN or Inf in label batch")
            continue

        optimizer.zero_grad()
        out = model(x)

        # Debug: Check for NaNs in output
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("Found NaN or Inf in model output")
            continue

        loss = loss_fn(out, y)

        # Debug: Check for NaNs in loss
        if torch.isnan(loss):
            print("Loss is NaN! Skipping this batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    if len(train_loader) == 0 or total == 0:
        return float('nan'), float('nan')

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy