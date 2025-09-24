from cupy import load
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli  # For binary actions
from scipy.sparse import load_npz
from torch.utils.data import DataLoader, TensorDataset

class ReflNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReflNetwork, self).__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Head 1: Classification (y)
        self.y_head = nn.Linear(hidden_dim, output_dim*3)
        # Head 2: REINFORCE (r)
        self.r_head = nn.Linear(hidden_dim, output_dim)  # Logits for binary actions
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        shared_out = self.shared(x)
        y_logits = self.y_head(shared_out)  # Classification logits
        y_logits = self.softmax(y_logits.view(y_logits.shape[0], -1, 3))
        r_logits = self.r_head(shared_out)  # Logits for binary actions
        return y_logits, r_logits

def compute_reward(x, y_logits, r_actions, KB):
    """Reward = count_nonzero(y_binary & r_actions - x_binary)"""
    # Step 1: Convert logits to binary masks
    y_binary = torch.abs(torch.argmax(y_logits, dim=-1) -1)
    
    # Step 2: Ensure x is binary (if not already)
    x_binary = (x > 0.5).float() if x.dtype != torch.long else x.float()

    r_binary = r_actions.to(bool)
    #x_binary = x_binary.to(bool)
    #y_binary = y_binary.to(bool)
    
    # Step 3: Compute reward
    reward = - torch.count_nonzero((y_binary == torch.clip(x_binary @ KB,0,1))[~r_binary], dim=-1).float()
    reward /= torch.count_nonzero(~r_binary)
    return reward

def train(
    model, 
    data_loader, 
    KB,
    num_epochs=10, 
    lr=1e-3, 
    gamma=0.99  # Discount factor for baseline
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    baseline = 0.0  # Moving average baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    KB = KB.to(device)

    for epoch in range(num_epochs):
        loss_y = 0.
        loss_r = 0.
        running_reward = 0.
        for x, y_true in data_loader:  # x: input, y_true: class labels
            x,y_true = x.to(device),y_true.to(device)

            # Forward pass
            y_logits, r_logits = model(x)

            # --- Loss 1: Cross-entropy for classification (y) ---
            ce_loss = F.cross_entropy(y_logits.view(-1,3), (y_true+1).view(-1))
            

            # --- Loss 2: REINFORCE for discrete actions (r) ---
            # Sample binary actions from Bernoulli distribution
            r_probs = torch.sigmoid(r_logits)
            dist = Bernoulli(r_probs)
            r_actions = dist.sample()  # Shape: (batch_size, output_dim)

            # Compute reward (L(y, r, x) = count_nonzero(y_binary & r_actions - x_binary))
            reward = compute_reward(x, y_logits, r_actions, KB)
            reward = reward.detach()  # Detach to avoid backprop through reward

            # Update baseline (exponential moving average)
            baseline = gamma * baseline + (1 - gamma) * reward.item()

            # REINFORCE loss
            log_probs = dist.log_prob(r_actions).sum(dim=1)  # Sum over output_dim
            r_loss = -torch.mean((reward - baseline) * log_probs)

            # Total loss
            total_loss = ce_loss + 0.1 * r_loss  # Scale REINFORCE loss to balance

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_y += ce_loss.item()
            loss_r += r_loss.item()
            running_reward += reward.item()

        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1}, CE Loss: {loss_y:.3f}, REINFORCE Loss: {loss_r:.3f}, Reward: {running_reward:.4f}")

if __name__ == '__main__':

    X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy')).float()
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_train.npy')).to(int)
    #Y_train = torch.abs(Y_train)

    KB = torch.tensor(load_npz('rules/regu_pos_clo.npz').toarray()).float()

    import pandas as pd
    label_set = pd.read_csv('dataset/ncbi-sra/label_set.csv')
    idx_list = list(label_set['matrix_idx'])
    KB = KB[:,idx_list]

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    hidden_dim = 128
    batch_size = 64
    
    # Initialize model
    model = ReflNetwork(input_dim, hidden_dim, output_dim)
    data_loader = DataLoader(TensorDataset(X_train,Y_train), batch_size=batch_size, shuffle=True)
    
    # Train
    train(model, data_loader, KB, num_epochs=5000)
