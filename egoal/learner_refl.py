import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from torch.distributions import Bernoulli

class ReflectNN(nn.Module):
    """ Network Structure of Base Learner with Reflect Output (RL) """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim:
            hidden_dim:
            output_dim:
        """
        super(ReflectNN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        #TODO use MLP temp, GNN embd?

        ' Head 1: Classification (y) '
        self.y_head = nn.Linear(hidden_dim, output_dim*3)
        self.softmax = nn.Softmax(dim=-1)

        ' Head 2: REINFORCE (r): Logits for binary actions '
        self.r_head = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)

        ' clf head '
        output_y = self.y_head(emb)
        output_y = self.softmax(output_y.view(output_y.shape[0], -1, 3))

        ' action head '
        output_r = self.r_head(emb)
        output_r = self.sigmoid(output_r)

        return output_y, output_r

    def predict(self,x):
        output_y, _ = self.forward(x)
        return torch.argmax(output_y, dim=-1) -1

    def reflection(self, x):
        _, output_r = self.forward(x)
        return output_r

class ReflectLearner():
    def __init__(self,
        KB: torch.Tensor,
        use_gpu = False,
        log_path = '',
    ) -> None:
        '''
        Args:
            KB:
            log_path (optional):
        '''

        ' 4639 genes of whole genome, 241 output genes '
        input_dim = 4639
        hidden_dim = 64
        output_dim = 623

        ' weight of classes for CE loss '
        self.clf_weight = torch.Tensor([.4,.2,.4])

        self.model = ReflectNN(input_dim, hidden_dim,  output_dim)
        self.KB = KB
        self.train_loader = None
        self.test_loader = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'cuda availability: {torch.cuda.is_available()}')
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = self.model.to(self.device)
            self.clf_weight = self.clf_weight.to(self.device)
            self.KB = self.KB.to(self.device)

        #if log_path != '':
        #    with open(log_path, 'w') as f:
        #        f.write('')
        self.log_path = log_path

    def consistency_reward(self, x, y_probs, r_binary):
        # TODO
        """Reward = count_nonzero(y_binary & r_actions - x_binary)"""
        # Step 1: Convert logits to binary masks
        y = torch.abs(torch.argmax(y_probs, dim=-1) -1)
        
        # Step 2: Ensure x is binary (if not already)
        #x_binary = (x > 0.5).float() if x.dtype != torch.long else x.float()
    
        r_binary = r_binary.to(bool)
        #x_binary = x_binary.to(bool)
        #y_binary = y_binary.to(bool)
        
        # Step 3: Compute reward
        reward = - torch.count_nonzero((y == torch.clamp(x @ self.KB,-1,1))[~r_binary], dim=-1).float()
        reward /= torch.count_nonzero(~r_binary)
        return reward
    
    def train(
        self, 
        epochs= 10, 
        lr= 1e-3, 
        gamma= 0.99
    ):
        '''
        Train the Clf + Refl Model
        Args:
            epochs: 
            lr:
            gamma: discount factor for RL baseline reward
        '''

        ''' Training loop '''
        criterion = nn.CrossEntropyLoss(weight=self.clf_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        baseline = 0.0
        for epoch in range(epochs):
            running_loss = 0.0
            for X_batch, Y_batch in self.train_loader:

                #if use_gpu: TODO
                #    X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
    
                output_y, output_r = self.model(X_batch)
    
                loss_y = criterion(output_y.view(-1,3), (Y_batch+1).view(-1))
                
    
                ' RL for discrete action opt '
                ' sample from Ber distribution, '
                dist = Bernoulli(output_r)
                r_actions = dist.sample()  # Shape: (batch_size, output_dim)
    
                reward = self.consistency_reward(X_batch, output_y, r_actions)
                reward = reward.detach()  # Detach to avoid backprop through reward
    
                # Update baseline (exponential moving average)
                baseline = gamma * baseline + (1 - gamma) * reward.item()
    
                # REINFORCE loss
                log_probs = dist.log_prob(r_actions).sum(dim=1)  # Sum over output_dim
                loss_r = -torch.mean((reward - baseline) * log_probs)
    
    
                ' backprop '
                total_loss = loss_y + 0.1 * loss_r  # Scale REINFORCE loss to balance
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
    
            if (epoch+1)%100 == 0:
                print(f"Epoch {epoch+1}, CE Loss: {ce_loss.item():.3f}, REINFORCE Loss: {r_loss.item():.3f}, Reward: {reward.item():.4f}")
