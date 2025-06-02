import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv
from torch_geometric.data import DataLoader
from torch.nn import Linear
from torch.utils.data import random_split
import os
from collections import Counter
from sklearn.metrics import classification_report
'Also bis jetzt macht es nur binary classification und noch keine Onset prediction, dass müssten wir allgemein noch adden und überlegen wie wir das machen, hab auch bis jetzt noch nicht geschaut obs klappt, aber wir haben jetzt schonmal das Grundgerüst für jeden schritt und müssen nur schaffen das es durchläuft :)'


# Build the GCN model: Ist jetzt n basic Ding können noch mit Layers und Nodes und allem so rumprobieren aber vllt mal als start hihi

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        
# Hier können wir mit den verschiedene Activation functions rumprobieren, relu, leaky relu, sigmoid,....
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x= F.relu(x)
        x = global_mean_pool(x, batch) # Hier können wir auch andere pooling methoden ausprobieren, global_max_pool zb
        x= self.lin(x)
        return x

# Jetzt starten wir das Training
# Initialisierung des Models mit genauen Daten
in_channels = 9
model = GCN(in_channels = in_channels, hidden_channels =32) # Man geht bei der Anzahl in 2^n Schritten also nächst größere wär 64, glaub hat was mit bits zu tun oder so
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005) # können auch anderen Optimizer ausprobieren und mit der learning rate auch rum tunen 
# Compute the Graph weights

weight_0 = 100 / (2 * 64)
weight_1 = 100 / (2 * 36)

class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)
#loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
loss_fn = torch.nn.CrossEntropyLoss() # Loss function fur binary classification

# Training Loop

def train_model(train_loader):
    model.train() # Model wird in training mode gestellt 
    total_loss = 0 
    for batch in train_loader: 
        optimizer.zero_grad() 
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
    return total_loss/ len(train_loader)

# Evaluate Model

def evaluate_model(loader):
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct = correct + (pred == batch.y).sum().item()
            y_true.extend(batch.y.tolist())
            y_pred.extend(pred.tolist())
    acc = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    return acc



