import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CNN_EEG(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(CNN_EEG, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels = 64, kernel_size = (3,5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, out_channels= 128, kernel_size = (3,5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_flattened_size(in_channels), 256) # Eingangsgröße muss angepasst werden
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    
    def _get_flattened_size(self, in_channels):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, 12, 9)  # 12x9 = your input shape
            x = self.pool1(F.relu(x))
            x = self.pool2(F.relu(x))
            return x.view(1, -1).shape[1]
        
    def forward(self,x):
        x = self.pool1(F.relu(x))
        x = self.pool2(F.relu(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
model = CNN_EEG(in_channels = 1, n_classes = 2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_fn = torch.nn.CrossEntropyLoss()

def train_model(model, train_loader, optimizer, loss_fn, device='cpu'):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    accuracy = correct / len(y_true)
    return accuracy, y_true, y_pred