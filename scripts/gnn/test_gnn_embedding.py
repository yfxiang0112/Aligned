import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import load
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

class FixedGraphGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, gnn_type='GCN'):
        """
        Args:
            num_nodes: Number of nodes in the fixed graph
            hidden_dim: Dimension of hidden layers
            gnn_type: Type of GNN layer ('GCN', 'GAT', or 'GraphSAGE')
        """
        super(FixedGraphGNN, self).__init__()
        self.num_nodes = num_nodes
        
        # One-hot embedding layer for node features
        self.embedding = nn.Linear(num_nodes, hidden_dim)
        
        # GNN layers
        if gnn_type == 'GCN':
            self.gnn1 = GCNConv(hidden_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'GAT':
            self.gnn1 = GATConv(hidden_dim, hidden_dim, heads=2)
            self.gnn2 = GATConv(2*hidden_dim, hidden_dim)  # Note dimension change for GAT
        elif gnn_type == 'GraphSAGE':
            self.gnn1 = SAGEConv(hidden_dim, hidden_dim)
            self.gnn2 = SAGEConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Classifier head for ternary classification
        self.classifier = nn.Linear(hidden_dim, 3)
        
        # Store edge_index as buffer if graph is fixed
        self.register_buffer('edge_index', None)
    
    def set_adjacency_matrix(self, adj_matrix):
        """Convert and store the adjacency matrix as edge index.
        Args:
            adj_matrix: [num_nodes, num_nodes] adjacency matrix
        """
        self.edge_index, _ = dense_to_sparse(adj_matrix)
    
    def forward(self, x_onehot):
        """
        Args:
            x_onehot: One-hot encoded node features [batch_size, num_nodes, num_nodes]
                     or [num_nodes, num_nodes] if single graph
        Returns:
            logits: Classification logits [batch_size, num_nodes, 3] or [num_nodes, 3]
        """
        if self.edge_index is None:
            raise RuntimeError("Adjacency matrix not set. Call set_adjacency_matrix() first.")

        print(x_onehot.shape)
        
        # Handle batch dimension if present
        if x_onehot.dim() == 3:
            batch_size = x_onehot.size(0)
            # Process each graph in batch
            x = self.embedding(x_onehot.float())  # [batch_size, num_nodes, hidden_dim]
            
            # Apply GNN layers
            x = x.transpose(0, 1)  # [num_nodes, batch_size, hidden_dim]
            x = F.relu(self.gnn1(x, self.edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.gnn2(x, self.edge_index))
            x = x.transpose(0, 1)  # [batch_size, num_nodes, hidden_dim]
            
            logits = self.classifier(x)
        else:
            # Single graph case
            x = self.embedding(x_onehot.float())  # [num_nodes, hidden_dim]
            x = F.relu(self.gnn1(x, self.edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.gnn2(x, self.edge_index))
            logits = self.classifier(x)
        
        return logits


if __name__ == '__main__':

    X = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype=torch.float32)
    Y = torch.tensor(np.load('dataset/precise1k/Y_train.npy'), dtype=int)
    train_dataset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    label_set = pd.read_csv('dataset/ncbi-sra/label_set.csv', index_col=0)
    label_idx = list(label_set['matrix_idx'])

    pos_regu = load_npz('rules/regu_pos.npz').toarray()[:,label_idx]
    neg_regu = load_npz('rules/regu_neg.npz').toarray()[:,label_idx]

    # Example parameters
    num_nodes = X.shape[1]
    hidden_dim = 256
    gnn_type = 'GCN'  # Choose from 'GCN', 'GAT', 'GraphSAGE'
    
    # Initialize model
    model = FixedGraphGNN(num_nodes, hidden_dim, gnn_type)
    
    adj_matrix = torch.clip(torch.tensor(pos_regu+neg_regu), 0,1)
    
    # Set the fixed graph structure
    model.set_adjacency_matrix(adj_matrix)
    
    # Example input (one-hot encoded node features)
    x_onehot = torch.eye(num_nodes)  # Identity matrix for one-hot
    
    # Forward pass
    logits = model(x_onehot)
    #print(logits.shape)  # Should be [10, 3] for 10 nodes and 3 classes

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Example training with batch processing
    def train(model, data_loader, epochs=100):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out.view(-1, 3), batch_y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(data_loader)}')

    # Example training loop
    for epoch in range(100):
        loss = train(model, train_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

