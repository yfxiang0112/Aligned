import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from tqdm import tqdm
import pandas as pd


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
        correct = 0
        total = 0
        with torch.no_grad():
            # confusion = np.zeros((3, 3), dtype=int)
            f1_micro2 = 0
            f1_macro2 = 0

            Y_test, Y_pred, Y_prob = [], [], []

            for X_batch, Y_batch in self.test_loader:
                outputs = self.model.predict(X_batch)

                Y_test.append(Y_batch)
                Y_pred.append(outputs)
                Y_prob.append(self.predict_prob(X_batch).max(dim=-1).values)
                total += Y_batch.size(0)
                # correct += (abs(outputs) == abs(Y_batch)).sum(dim=0)
                correct += (outputs == Y_batch).sum(dim=0)
                # print(outputs)
                # outputs = torch.where(outputs==-1, 2, outputs)
                # Y_batch = torch.where(Y_batch==-1, 2, Y_batch)

                # NOTE temp test
                # print(outputs, np.count_nonzero(outputs))
                # print(Y_batch, np.count_nonzero(Y_batch))

            Y_test = torch.concat(Y_test, dim=0)
            Y_pred = torch.concat(Y_pred, dim=0)
            Y_prob = torch.concat(Y_prob, dim=0)

            ''' compute total confusion matrix '''
            flat_y_t = Y_test.flatten()
            flat_y_p = Y_pred.flatten()
            confusion = confusion_matrix(flat_y_t, flat_y_p, labels=[-1, 0, 1])
            confusion = confusion / confusion.sum().sum()
            # micro on labels, macro on classes
            f1_macro = f1_score(flat_y_t, flat_y_p, average='macro')
            # micro on labels, micro on classes
            f1_micro = f1_score(flat_y_t, flat_y_p, average='micro')

            ' compute weighted f1 by ground truth proportion '
            weights = [1/(torch.sum(flat_y_t == -1).item() + 1e-6),
                       1/(torch.sum(flat_y_t == 0).item() + 1e-6),
                       1/(torch.sum(flat_y_t == 1).item() + 1e-6)]
            weights = torch.Tensor(weights) / sum(weights)
            f1_class = f1_score(flat_y_t, flat_y_p, average=None)
            f1_weighted = sum([f1*w for f1, w in zip(f1_class, weights)])
            # f1_weighted = f1_class[0]*weights[0] + f1_class[2]*weights[2]

            for label_idx in range(Y_test.shape[1]):
                # macro on labels, macro on classes
                f1_macro2 += f1_score(Y_test[:, label_idx],
                                      Y_pred[:, label_idx], average='macro')
                # macro on labels, macro on classes
                f1_micro2 += f1_score(Y_test[:, label_idx],
                                      Y_pred[:, label_idx], average='micro')

                #    print(score)
            f1_macro2 /= Y_test.shape[1]
            f1_micro2 /= Y_test.shape[1]

            # Y_pred = torch.where(Y_pred==2, -1, Y_pred)
            # Y_test = torch.where(Y_test==2, -1, Y_test)

            ''' compute acc & confusion matrix on each gene '''
            per_label_accuracy = correct / total
            # f1 /= Y_test.shape[1]
            if self.log_path != '':
                f1_ = []
                csv_file_name = self.log_path.replace('.txt', '.csv')
                csv_data = {
                    'label': [f'{i:8}' for i in range(len(per_label_accuracy))],
                    'accuracy': [f'{acc * 100:7.2f}%' for acc in per_label_accuracy],
                }
                with open(self.log_path, 'a') as f:
                    f.write('label ')
                    for i in range(len(per_label_accuracy)):
                        f.write(f'{i:8}\t')
                    f.write('\n   acc ')
                    for acc in per_label_accuracy:
                        f.write(f'{acc * 100:7.2f}%\t')
                    f.write('\n    f1 ')
                    for label_idx in range(Y_test.shape[1]):
                        f1_score_ = f1_score(
                            Y_test[:, label_idx], Y_pred[:, label_idx], average='macro')
                        f1_.append(f1_score_)
                        f.write(
                            f"{f1_score_:8.4f}\t")
                    csv_data['f1'] = f1_
                    # f.write('\n---- data ----')

                    for data_idx in range(Y_test.shape[0]):
                        f.write(f'\npred{data_idx:2} ')
                        for y_pred in Y_pred[data_idx]:
                            f.write(f'{y_pred:8}\t')
                        f.write(f'\nprob{data_idx:2} ')
                        for y_prob in Y_prob[data_idx]:
                            f.write(f'{y_prob:8.2f}\t')
                        f.write(f'\ntest{data_idx:2} ')
                        for y_test in Y_test[data_idx]:
                            f.write(f'{y_test:8}\t')
                        csv_data[f'pred{data_idx:2}'] = Y_pred[data_idx]
                        csv_data[f'prob{data_idx:2}'] = Y_prob[data_idx]
                        csv_data[f'test{data_idx:2}'] = Y_test[data_idx]

                    f.write(f'\n------\nconfusion matrix:\n{confusion}\n')
                    f.write(f'macro f1: {f1_macro}\n')
                    f.write(f'micro f1: {f1_micro}\n')
                    # f.write(f'macro f1 2: {f1_macro2}\n')
                    # f.write(f'micro f1 2: {f1_micro2}\n')
                    f.write(f'weighted f1: {f1_weighted}\n')
                    f.write(f'class -1 f1: {f1_class[0]}\n')
                    f.write(f'class  0 f1: {f1_class[1]}\n')
                    f.write(f'class  1 f1: {f1_class[2]}\n')
                    f.write(
                        f'average label-wise acc: {np.mean(per_label_accuracy.numpy())*100:.2f}%\n')
                pd.DataFrame(csv_data).to_csv(
                    csv_file_name, index=False)

            else:
                print(
                    f'Average Per-label Acc: {np.mean(per_label_accuracy.numpy())*100:.2f}%\n')

            return f1_macro

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
