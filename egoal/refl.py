import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Policy Network (outputs Bernoulli probabilities for each dimension)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)  # Probabilities for each y_i (Bernoulli)

class Reflector():
    def __init__(self) -> None:
        self.label_size = 623 * 2
        self.input_size = 256
        self.policy_net = PolicyNetwork(input_dim=self.input_size, output_dim=self.label_size)
        self.policy_net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Sample a boolean vector y and compute its log probability
    def sample_action(self, init_input = None):
        if init_input == None:
            init_input = torch.zeros(self.label_size)

        probs =self.policy_net(init_input)
        y = torch.bernoulli(probs)  # Binary vector (0 or 1)
        log_probs = torch.sum(torch.log(probs * y + (1 - probs) * (1 - y)))  # Log prob of y
        return y, log_probs, probs
    
    # REINFORCE update with entropy regularization and reward normalization
    def reinforce_update(self, optimizer, y, log_prob, reward, probs, entropy_coef=0.01):
        # Entropy regularization (encourage exploration)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        # Policy gradient loss (maximize reward)
        loss = -log_prob * reward - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Training loop
    def train(self, L, x=None, episodes=1000, lr=0.01, gamma=0.99, silent=True):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        baseline = None  # For reward normalization
        baseline_decay = 0.9  # Exponential moving average decay
        
        for episode in range(episodes):
            y, log_prob, probs = self.sample_action(x)
            y_np = y.detach().numpy()  # Convert to numpy for L(y)
            reward = -L(y_np)  # Reward = -Loss (to minimize L)
            
            # Update baseline for reward normalization (reduces variance)
            if baseline is None:
                baseline = reward
            else:
                baseline = baseline_decay * baseline + (1 - baseline_decay) * reward
            
            # Normalized reward (advantage)
            normalized_reward = reward - baseline
            
            # Update policy
            self.reinforce_update(optimizer, y, log_prob, normalized_reward, probs)
            
            # Log progress
            if episode % 100 == 0 and not silent:
                print(f"Episode {episode}, Loss: {-reward:.4f}, Baseline: {baseline:.4f}")

    def predict(self, x):
        return torch.bernoulli(self.policy_net.forward(x))
    
    # Example loss function (minimize number of 1s)
def L(y):
    return np.sum(y)  # Example: Minimize the number of 1s

if __name__ == "__main__":
    #policy_net = PolicyNetwork()
    refl= Reflector()
    refl.train(L, episodes=1000)
