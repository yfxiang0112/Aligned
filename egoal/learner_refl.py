import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from torch.distributions import Bernoulli

from egoal.reasoner import RegualtoryKB

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
        self.relu = nn.ReLU()

        ' Head 1: Classification (y) '
        self.y_head = nn.Linear(hidden_dim, output_dim*3)
        self.softmax = nn.Softmax(dim=-1)

        ' Head 2: REINFORCE (r): Logits for binary actions '
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.r_head = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)

        ' clf head '
        output_y = self.y_head(emb)
        output_y = self.softmax(output_y.view(output_y.shape[0], -1, 3))

        ' action head '
        #output_r = self.r_head(self.relu(self.fc(emb)))
        output_r = self.r_head(emb)
        output_r = self.sigmoid(output_r)

        return output_y, output_r

    def predict(self,x):
        output_y, _ = self.forward(x)
        return torch.argmax(output_y, dim=-1) -1

    def reflection(self, x):
        _, output_r = self.forward(x)
        return torch.round(output_r)

class ReflectLearner():
    def __init__(self,
        input_dim,
        output_dim,
        hidden_dim = 64,
        device = 'cpu',
        log_path = '',
    ) -> None:
        '''
        Args:
            input_dim:
            output_dim:
            hidden_dim:
            device:
            log_path (optional):
        '''

        ' 4639 genes of whole genome, 241 output genes '
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ' weight of classes for CE loss '
        self.clf_weight = torch.Tensor([.4,.2,.4])

        self.model = ReflectNN(self.input_dim, self.hidden_dim,  self.output_dim)
        self.train_loader = None
        self.test_loader = None

        self.device = device
        print(f'cuda availability: {torch.cuda.is_available()}')
        if self.device != 'cpu':
            self.model = self.model.to(self.device)
            self.clf_weight = self.clf_weight.to(self.device)

        self.log_path = log_path


    def consistency_reward(self,
                           KB: RegualtoryKB,
                           x: torch.Tensor,
                           y_probs: torch.Tensor,
                           r_binary: torch.Tensor,
                           label_weight = None | torch.Tensor,
                           th = .3):
        '''
        Args:
            KB:
            x:
            y_probs:
            r_binary:
            label_weight:
            th:

        Return Value:

        '''

        y = torch.argmax(y_probs, dim=-1) -1

        violated = KB.violated(Y=y, X=x, mask=~(r_binary.bool()))

        weighted_restriction = torch.sum(torch.clamp(
            torch.sign(r_binary - .5) * (-label_weight), min=0))\
                    if label_weight != None else 0

        total = r_binary.shape[0] * r_binary.shape[1]
        len_restriction = torch.max(torch.count_nonzero(r_binary) - th * total, other=torch.tensor(0))

        return - violated - 2*weighted_restriction - len_restriction
    

    def load_data(self,
                  X_train: None | torch.Tensor,
                  Y_train: None | torch.Tensor,
                  X_test: torch.Tensor,
                  Y_test: torch.Tensor,
                  update_weight = False,
                  batch_size=64):
        '''
        define train & test data loader

        Args:
            X_train:
            Y_train:
            X_test:
            Y_test:
            update_weight:
            batch_size=64:
        '''

        assert len(X_test) > 0
        assert len(Y_test) > 0
        test_dataset = TensorDataset(X_test, Y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if X_train != None and Y_train != None:
            assert len(X_train) > 0
            assert len(Y_train) > 0
            train_dataset = TensorDataset(X_train, Y_train)
                    
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            ' reset classification loss weight with new Y_train '
            if update_weight:
                flat_y = Y_train.flatten()
                weights = [1/(torch.sum(flat_y==-1).item() + 1e-6),
                           1/(torch.sum(flat_y==0).item() + 1e-6),
                           1/(torch.sum(flat_y==1).item() + 1e-6)]

                self.clf_weight = torch.Tensor(weights) / sum(weights)
                if self.device != 'cpu':
                    self.clf_weight = self.clf_weight.to(self.device)


    def train(
        self, 
        KB: RegualtoryKB,
        label_weight= None | torch.Tensor,
        epochs= 10, 
        reinforce_epochs= 100,
        C= 1,
        lr= 1e-3, 
        gamma= 0.95,
        verbose= False
    ):
        '''
        Train the Clf + Refl Model
        Args:
            KB:
            label_weight:
            epochs: 
            reinforce_epochs:
            C:
            lr:
            gamma:          discount factor for RL baseline reward
            verbose:
        '''

        ''' Training loop '''
        criterion = nn.CrossEntropyLoss(weight=self.clf_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        baseline = 0.

        for epoch in range(epochs * reinforce_epochs):

            ' get the global reward '
            r_actions_batch = []
            #violated, r_nonzero = 0., 0.
            reward = 0.
            for X_batch, _ in self.train_loader:

                output_y, output_r = self.model(X_batch)

                output_r = torch.clamp(output_r.detach(), .2, .8)
                # NOTE test

                dist = Bernoulli(output_r)
                r_actions = dist.sample()  # Shape: (batch_size, output_dim)
                r_actions_batch.append(r_actions)

                reward += self.consistency_reward(KB, X_batch, output_y, r_actions, label_weight).detach().item()
                #r_nonzero += torch.count_nonzero(1-r_actions).detach().item()

            #reward = violated / (r_nonzero + 1e-6)
            #reward = violated

            ' opt local loss_y & global loss_r '
            loss_y, loss_r = 0.,0.
            for (X_batch, Y_batch), r_actions in zip(self.train_loader, r_actions_batch):
                output_y, output_r = self.model(X_batch)
                Y_batch = Y_batch.to(int)+1

                ' CE loss '
                loss_y += criterion(output_y.view(-1,3), Y_batch.view(-1))

                ' RL for discrete action opt '
                ' sample from Ber distribution, '
                dist = Bernoulli(output_r)

                ' Update baseline (exponential moving average) '
                baseline = gamma * baseline + (1 - gamma) * reward

                # REINFORCE loss
                log_probs = dist.log_prob(r_actions).sum(dim=1)
                loss_r += -torch.mean((reward - baseline) * log_probs)
    
    
            ' backprop '
            if epoch % reinforce_epochs == 0:
                C1, C2 = 1, C
            else:
                C1, C2 = 0, C

            total_loss = C1 * loss_y + C2 * loss_r  # Scale REINFORCE loss to balance
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch+1)%100 == 0 and verbose:
                print(f"Epoch {epoch+1}, Total loss: {total_loss.item():.4f}, CE loss: {loss_y.item():.4f}, RL loss: {loss_r.item():.4f}, Reward: {reward:.4f}")

                #NOTE tmp
                for X_batch, _ in self.train_loader:
                    output_y, output_r = self.model(X_batch)
                    y = torch.argmax(output_y, dim=-1) -1
                    r = torch.round(output_r)
                    print(f'    full cols: {torch.count_nonzero(torch.sum(r,dim=0)==len(r))}, non-full cols: {torch.count_nonzero((torch.sum(r,dim=0)<len(r)) & (torch.sum(r,dim=0)>0))}')
                    #print(f'r={output_r}')
                    
                    labels = [1, 2, 23, 82, 83, 84, 97, 107, 159, 160, 161, 162, 163, 243, 246, 263, 264, 265, 266, 267, 268, 269, 272, 284, 286, 301, 302, 303, 328, 337, 338, 342, 343, 348, 353, 362, 363, 418, 432, 436, 443, 444, 448, 458, 471, 501, 549, 550, 558, 561, 562, 570, 579, 584, 585, 586, 587, 588, 589, 621, 622, 632, 633, 634, 641, 661, 683, 705, 719, 720, 734, 748, 752, 753, 762, 763, 764, 765, 766, 799, 833, 904, 925, 934, 945, 950, 951, 952, 953, 954, 955, 965, 997, 998, 999, 1003, 1011, 1012, 1013, 1029, 1030, 1031, 1032, 1043, 1052, 1066, 1071, 1091, 1094, 1095, 1132, 1133, 1136, 1140, 1141, 1142, 1175, 1178, 1183, 1204, 1209, 1213, 1215, 1251, 1252, 1262, 1320, 1321, 1322, 1339, 1342, 1347, 1348, 1349, 1366, 1381, 1382, 1388, 1389, 1391, 1392, 1402, 1403, 1429, 1430, 1438, 1439, 1442, 1443, 1452, 1462, 1474, 1475, 1497, 1498, 1507]
                    r_idx = torch.nonzero(torch.sum(r,dim=0)).squeeze(-1).cpu().detach().numpy().tolist()
                    print(f'   r - labels: {len(set(r_idx)-set(labels))}, labels - r: {len(set(labels) - set(r_idx))}')
                    #print(f'    r-labels: {output_r[0,list(set(r_idx)-set(labels))]}\n    labels-r: {output_r[0,list(set(labels)-set(r_idx))]}')
                    
                    violated = KB.violated(Y=y, X=X_batch, mask=~(r.bool()))
                    weighted_restriction = torch.sum(torch.clamp(
                        torch.sign(r- .5) * (-label_weight), min=0))\
                                if label_weight != None else 0
                    total = r.shape[0] * r.shape[1]
                    len_restriction = torch.max(torch.count_nonzero(r) - .3 * total, other=torch.tensor(0))
                    print(f'    violated: {violated}, weighted: {weighted_restriction}, len: {len_restriction}, nonzero: {torch.count_nonzero(r) / total}')
                    break


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

            if self.device != 'cpu':
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

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def predict(self, x: torch.Tensor):
        return self.model.predict(x)

    def predict_prob(self, x: torch.Tensor):
        outputs, _ = self.model(x)
        return outputs

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype=torch.float32)
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_label.npy'), dtype=int)
    X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype=torch.float32)
    Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_label.npy'), dtype=int)

    import pandas as pd
    label_set = pd.read_csv('dataset/label_set_iml.csv')
    idx_list_p1k = list(label_set['precise1k_idx'])
    idx_list_sra = list(label_set['matrix_idx'])

    Y_train = Y_train[:,idx_list_p1k]
    Y_test = Y_test[:,idx_list_sra]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)



    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    hidden_dim = 128
    batch_size = 64
    

    # Initialize model
    #data_loader = DataLoader(TensorDataset(X_train,Y_train), batch_size=batch_size, shuffle=True)
    #learner.train_loader = data_loader

    # Train
    learner = ReflectLearner(input_dim=X_train.shape[1], output_dim=Y_train.shape[1], device=device, log_path='log.txt')
    regulatory_kb = RegualtoryKB(pos_trn_pth= 'rules/regu_pos.npz', neg_trn_pth='rules/regu_neg.npz', output_idx_list=idx_list_sra, device=device)
    regulatory_kb.closure_(T=5, closure_type='weighted')

    learner.load_data(X_train, Y_train, X_test, Y_test, batch_size=batch_size)
    print(learner.eval())
    learner.train(KB= regulatory_kb, epochs=50000, lr=1e-4)
    print(learner.eval())
