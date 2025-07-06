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

    def predict_prob(self,x):
        output_y, _ = self.forward(x)
        return output_y

    def reflection(self, x):
        _, output_r = self.forward(x)
        return torch.round(output_r)

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

        self.log_path = log_path


    def consistency_reward(self, x, y_probs, r_binary):
        # TODO
        """Reward = count_nonzero(y_binary & r_actions - x_binary)"""
        y = torch.abs(torch.argmax(y_probs, dim=-1) -1)
        r_binary = r_binary.to(bool)
        
        reward = - torch.count_nonzero((y == torch.clamp(x @ self.KB,-1,1))[~r_binary], dim=-1).float()
        #reward /= torch.count_nonzero(~r_binary)
        return reward
    

    def load_data(self, X_train, Y_train, X_test, Y_test, batch_size=64):
        ''' define train & test data loader '''
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(Y_train) > 0
        assert len(Y_test) > 0
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        ''' reset classification loss weight with new Y_train '''
        flat_y = Y_train.flatten()
        weights = [1/(torch.sum(flat_y==-1).item() + 1e-6),
                   1/(torch.sum(flat_y==0).item() + 1e-6),
                   1/(torch.sum(flat_y==1).item() + 1e-6)]
        #self.clf_weight = torch.Tensor(weights) / sum(weights)
        #if self.use_gpu:
        #    self.clf_weight = self.clf_weight.to(self.device)


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

        baseline = 0.

        for epoch in range(epochs):

            ' get the global reward '
            r_actions_batch = []
            violated, r_nonzero = 0., 0.
            for X_batch, _ in self.train_loader:

                output_y, output_r = self.model(X_batch)

                dist = Bernoulli(output_r)
                r_actions = dist.sample()  # Shape: (batch_size, output_dim)
                r_actions_batch.append(r_actions)

                violated += self.consistency_reward(X_batch, output_y, r_actions).detach().item()
                r_nonzero += torch.count_nonzero(1-r_actions).detach().item()

            reward = violated / (r_nonzero + 1e-6)

            ' opt local loss_y & global loss_r '
            loss_y, loss_r = 0.,0.
            for (X_batch, Y_batch), r_actions in zip(self.train_loader, r_actions_batch):
                output_y, output_r = self.model(X_batch)

                ' CE loss '
                loss_y += criterion(output_y.view(-1,3), (Y_batch+1).view(-1))

                ' RL for discrete action opt '
                ' sample from Ber distribution, '
                dist = Bernoulli(output_r)

                ' Update baseline (exponential moving average) '
                baseline = gamma * baseline + (1 - gamma) * reward

                # REINFORCE loss
                log_probs = dist.log_prob(r_actions).sum(dim=1)
                loss_r += -torch.mean((reward - baseline) * log_probs)
    
    
            ' backprop '
            total_loss = loss_y + 0.1 * loss_r  # Scale REINFORCE loss to balance
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch+1)%100 == 0:
                print(f"Epoch {epoch+1}, Total loss: {total_loss.item():.4f}, CE loss: {loss_y.item():.4f}, RL loss: {loss_r.item():.4f}, Reward: {reward:.4f}")


    #########################################################################

    def eval(self):
        assert self.test_loader != None

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            f1_micro2 = 0
            f1_macro2 = 0

            Y_test, Y_pred, Y_prob = [],[],[]

            for X_batch, Y_batch in self.test_loader:
                outputs = self.model.predict(X_batch)

                Y_test.append(Y_batch)
                Y_pred.append(outputs)
            #    Y_prob.append(self.predict_prob(X_batch).max(dim=-1).values)
                total += Y_batch.size(0)
                correct += (outputs == Y_batch).sum(dim=0)

            Y_test = torch.concat(Y_test, dim=0)
            Y_pred = torch.concat(Y_pred, dim=0)

            if self.use_gpu:
                Y_test = Y_test.cpu()
                Y_pred = Y_pred.cpu()
                correct = correct.cpu()
            #Y_prob = torch.concat(Y_prob, dim=0)

            ''' compute total confusion matrix '''
            flat_y_t = Y_test.flatten()
            flat_y_p = Y_pred.flatten()
            confusion = confusion_matrix(flat_y_t, flat_y_p, labels=[-1, 0,1])
            confusion = confusion / confusion.sum().sum()
            f1_macro = f1_score(flat_y_t, flat_y_p, average='macro') # micro on labels, macro on classes
            f1_micro = f1_score(flat_y_t, flat_y_p, average='micro') # micro on labels, micro on classes

            ' compute weighted f1 by ground truth proportion '
            weights = [1/(torch.sum(flat_y_t==-1).item() + 1e-6),
                       1/(torch.sum(flat_y_t==0).item() + 1e-6),
                       1/(torch.sum(flat_y_t==1).item() + 1e-6)]
            weights = torch.Tensor(weights) / sum(weights)
            f1_class = f1_score(flat_y_t, flat_y_p, average=None)
            f1_weighted = sum([f1*w for f1,w in zip(f1_class,weights)])
            #f1_weighted = f1_class[0]*weights[0] + f1_class[2]*weights[2]

            for label_idx in range(Y_test.shape[1]):
                f1_macro2 += f1_score(Y_test[:,label_idx], Y_pred[:,label_idx], average='macro') # macro on labels, macro on classes
                f1_micro2 += f1_score(Y_test[:,label_idx], Y_pred[:,label_idx], average='micro') # macro on labels, macro on classes
                    
            f1_macro2 /= Y_test.shape[1]
            f1_micro2 /= Y_test.shape[1]

            ''' compute acc & confusion matrix on each gene '''
            per_label_accuracy = correct / total
            #f1 /= Y_test.shape[1]
            if self.log_path != '':
                with open(self.log_path,'a') as f:
                    f.write('label ')
                    for i in range(len(per_label_accuracy)):
                        f.write(f'{i:8}\t')
                    f.write('\n   acc ')
                    for acc in per_label_accuracy:
                        f.write(f'{acc * 100:7.2f}%\t')
                    f.write('\n    f1 ')
                    for label_idx in range(Y_test.shape[1]):
                        f.write(f"{f1_score(Y_test[:,label_idx], Y_pred[:,label_idx], average='macro'):8.4f}\t")

                    #for data_idx in range(Y_test.shape[0]):
                    #    f.write(f'\npred{data_idx:2} ')
                    #    for y_pred in Y_pred[data_idx]:
                    #        f.write(f'{y_pred:8}\t')
                    #    f.write(f'\nprob{data_idx:2} ')
                    #    for y_prob in Y_prob[data_idx]:
                    #        f.write(f'{y_prob:8.2f}\t')
                    #    f.write(f'\ntest{data_idx:2} ')
                    #    for y_test in Y_test[data_idx]:
                    #        f.write(f'{y_test:8}\t')

                    f.write(f'\n------\nconfusion matrix:\n{confusion}\n')
                    f.write(f'macro f1: {f1_macro}\n')
                    f.write(f'micro f1: {f1_micro}\n')
                    f.write(f'weighted f1: {f1_weighted}\n')
                    f.write(f'class -1 f1: {f1_class[0]}\n')
                    f.write(f'class  0 f1: {f1_class[1]}\n')
                    f.write(f'class  1 f1: {f1_class[2]}\n')
                    f.write(f'average label-wise acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')
            else:
                print(f'Average Per-label Acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')

            return f1_macro



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype=torch.float32)
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_train.npy'), dtype=int)
    X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype=torch.float32)
    Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_train.npy'), dtype=int)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)


    from scipy.sparse import load_npz
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
    #data_loader = DataLoader(TensorDataset(X_train,Y_train), batch_size=batch_size, shuffle=True)
    #learner.train_loader = data_loader

    # Train
    learner = ReflectLearner(KB, use_gpu=True, log_path='log.txt')
    learner.load_data(X_train, Y_train, X_test, Y_test, batch_size=batch_size)
    print(learner.eval())
    learner.train(epochs=10000, lr=1e-4)
    print(learner.eval())
