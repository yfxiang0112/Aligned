import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
num_nodes = 4000
input_dim = 4000
output_dim = 600
hidden_dim = 256
epochs = 100
learning_rate = 0.01
test_size = 0.2

# Generate synthetic data for demonstration
# In practice, you would load your real data here
adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()  # Binary adjacency matrix
node_features = torch.randint(0, 2, (num_nodes//2, input_dim)).float()  # Binary node features
labels = torch.randint(-1, 2, (num_nodes, output_dim)).float()  # -1/0/1 labels for 600 classes

# Only a subset of nodes have labels (assuming 80% are labeled)
mask = torch.rand(num_nodes) < 0.8
labels[~mask] = float('nan')  # Mark unlabeled nodes with NaN

# Convert adjacency matrix to edge index format expected by PyG
edge_index = (adj_matrix > 0).nonzero().t().contiguous()

# Create PyG Data object
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Split nodes into train and test sets
train_indices, test_indices = train_test_split(
    torch.arange(num_nodes)[mask],  # Only use nodes that have labels
    test_size=test_size,
    random_state=42
)

# Create masks for training and testing
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True

data.train_mask = train_mask
data.test_mask = test_mask

# Define GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.fc(x)
        return x

# Initialize model, optimizer, and loss function
model = GNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    
    # Only compute loss on labeled training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = torch.sigmoid(out)  # Convert to probabilities
        
        # Threshold for -1/0/1 classification
        final_pred = torch.zeros_like(pred)
        final_pred[pred > 0.66] = 1    # High confidence positive
        final_pred[pred < 0.33] = -1    # High confidence negative
        # 0.33-0.66 remains 0 (uncertain)
        
        # Only evaluate on masked nodes
        true_labels = data.y[mask]
        pred_labels = final_pred[mask]
        
        # Filter out NaN labels (unlabeled nodes)
        valid_indices = ~torch.isnan(true_labels).any(dim=1)
        true_labels = true_labels[valid_indices]
        pred_labels = pred_labels[valid_indices]
        
        # Calculate accuracy (exact match)
        accuracy = (true_labels == pred_labels).all(dim=1).float().mean().item()
        
        # Calculate F1 score (micro-averaged)
        # Flatten all labels for F1 calculation
        f1 = f1_score(true_labels.view(-1).cpu().numpy(), 
                     pred_labels.view(-1).cpu().numpy(), 
                     average='micro')
        
        return accuracy, f1

# Training loop
for epoch in range(1, epochs + 1):
    loss = train()
    train_acc, train_f1 = evaluate(data.train_mask)
    test_acc, test_f1 = evaluate(data.test_mask)
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}')

# Final evaluation
model.eval()
final_train_acc, final_train_f1 = evaluate(data.train_mask)
final_test_acc, final_test_f1 = evaluate(data.test_mask)

print('\nFinal Results:')
print(f'Train Accuracy: {final_train_acc:.4f}')
print(f'Test Accuracy: {final_test_acc:.4f}')
print(f'Train F1 Score: {final_train_f1:.4f}')
print(f'Test F1 Score: {final_test_f1:.4f}')
