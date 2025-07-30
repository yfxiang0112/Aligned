import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import pandas as pd


from egoal.utils import eval_log

class BaseLearnerNN(nn.Module):
    ''' network struct of base learner '''

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(BaseLearnerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, output_dim * 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def embedding(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x.view(x.shape[0], -1, 3)

    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, dim=-1) - 1


class BaseLearner():
    def __init__(self, log_path='') -> None:
        ''' 4639 genes of whole genome '''
        input_dim = 4639

        hidden1_dim = 4096
        hidden2_dim = 1024
        hidden3_dim = 256

        ''' 241 output genes '''
        output_dim = 623

        ''' weight of classes for CE loss '''
        self.clf_weight = torch.Tensor([.4, .2, .4])

        self.model = BaseLearnerNN(
            input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim)
        self.train_loader = None
        self.test_loader = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())

        # if log_path != '':
        #    with open(log_path, 'w') as f:
        #        f.write('')
        self.log_path = log_path

    def split_data(self, X, Y, test_size=0.2, random_seed=42, test_indices=[]):
        '''
        random shuffle if test indices is not provided,
        set given indices as test set else
        '''
        n_samples = X.shape[0]

        if test_indices == []:
            ''' get shuffled index array '''
            np.random.seed(random_seed)
            indices = np.random.permutation(n_samples)

            ''' Compute split index '''
            split_idx = int(n_samples * (1 - test_size))
            train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        else:
            train_indices = [i for i in range(
                n_samples) if not (i in test_indices)]

        X_train, X_test = torch.tensor(X[train_indices], dtype=torch.float32), \
            torch.tensor(X[test_indices], dtype=torch.float32)
        Y_train, Y_test = torch.tensor(Y[train_indices], dtype=int), \
            torch.tensor(Y[test_indices], dtype=int)
        return X_train, X_test, Y_train, Y_test

    def load_data(self, X_train, Y_train, X_test, Y_test, batch_size=64):
        ''' define train & test data loader '''
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(Y_train) > 0
        assert len(Y_test) > 0
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        ''' reset classification loss weight with new Y_train '''
        flat_y = Y_train.flatten()
        # weights = [torch.sum(Y_train==-1, axis=0),
        #           torch.sum(Y_train==0, axis=0),
        #           torch.sum(Y_train==1, axis=0)]
        weights = [1/(torch.sum(flat_y == -1).item() + 1e-6),
                   1/(torch.sum(flat_y == 0).item() + 1e-6),
                   1/(torch.sum(flat_y == 1).item() + 1e-6)]
        # weights = torch.stack(weights)
        # weights = 1/ (weights + 1e-4)
        self.clf_weight = torch.Tensor(weights) / sum(weights)
        # print(weights)
        # print(weights.shape)

    def train(self, epochs: int, mask=None, use_gpu=False, lr=0.001):
        assert self.train_loader != None
        # if self.log_path != '':
        #    with open(self.log_path,'a') as f:
        #        f.write('---------- Train ----------\n')

        # print(f'weight = {self.clf_weight}')

        ' load model to gpu '
        if use_gpu:
            self.model = self.model.to(self.device)
            self.clf_weight = self.clf_weight.to(self.device)

        ''' Training loop '''
        criterion = nn.CrossEntropyLoss(weight=self.clf_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in tqdm(range(epochs)):
            running_loss = 0.0
            for X_batch, Y_batch in self.train_loader:
                if use_gpu:
                    X_batch, Y_batch = X_batch.to(
                        self.device), Y_batch.to(self.device)

                ' Forward pass '
                outputs = self.model(X_batch)
                if mask:
                    if use_gpu:
                        mask = mask.to(self.device)
                    selected_outputs = torch.masked_select(outputs, mask)
                    selected_targets = torch.masked_select((Y_batch+1), mask)
                    loss = criterion(selected_outputs.view(-1, 3),
                                     selected_targets.view(-1))
                else:
                    loss = criterion(outputs.view(-1, 3), (Y_batch+1).view(-1))

                ''' Backward pass and optimization '''
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # if self.log_path != '':
            #    with open(self.log_path,'a') as f:
            #        f.write(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.train_loader):.4f}\n")
        ' move model back to cpu '
        if use_gpu:
            self.model = self.model.to("cpu")

    def eval(self):
        assert self.test_loader != None
        # if self.log_path != '':
        #    with open(self.log_path,'a') as f:
        #        f.write('---------- Eval ----------\n')

        self.model.eval()
        with torch.no_grad():

            Y_test, Y_pred, Y_prob = [], [], []

            for X_batch, Y_batch in self.test_loader:
                outputs = self.model.predict(X_batch)

                Y_test.append(Y_batch)
                Y_pred.append(outputs)
                Y_prob.append(self.predict_prob(X_batch).max(dim=-1).values)

            Y_test = torch.concat(Y_test, dim=0)
            Y_pred = torch.concat(Y_pred, dim=0)
            Y_prob = torch.concat(Y_prob, dim=0)

            f1 = eval_log(Y_test, Y_pred, self.log_path, Y_prob=Y_prob)

            return f1

    def predict(self, X):
        if type(X) != torch.Tensor:
            X = torch.Tensor(X)
        return self.model.predict(X)

    def predict_prob(self, X):
        if type(X) != torch.Tensor:
            X = torch.Tensor(X)
        outputs = torch.exp(self.model.forward(X))
        return outputs / torch.sum(outputs, dim=-1, keepdim=True)

    def save(self, path):
        torch.save(self.model.state_dict(), path)


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor(
        np.load('dataset/precise1k/X_label.npy'), dtype=torch.float32)
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_train.npy'), dtype=int)
    X_test = torch.tensor(
        np.load('dataset/ncbi-sra/X_label.npy'), dtype=torch.float32)
    Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_train.npy'), dtype=int)

    # test_idx = torch.tensor(np.random.choice([True, False], size=len(X_train), p=[.2, .8]))
    # X_test, Y_test = X_train[test_idx], Y_train[test_idx]
    # X_train, Y_train = X_train[~test_idx], Y_train[~test_idx]

    ''' add logger & training '''
    learner = BaseLearner(log_path='log.txt')
    learner.load_data(X_train, Y_train, X_test, Y_test)
    print(learner.eval())
    learner.train(epochs=50, lr=1e-3)
    print(learner.eval())
    #learner.save('models/model_prec1k.pt')
