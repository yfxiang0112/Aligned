import torch
import torch.nn as nn
from torch_geometric.nn import SGConv
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from torch.distributions import Bernoulli
from torch_geometric.utils import dense_to_sparse
from scipy.sparse import load_npz

from egoal.reasoner import RegulatoryKB

class ReflectMLP(nn.Module):
    """ Network Structure of Base Learner with Reflect Output (RL) """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 discretized=True):
        """
        Args:
            input_dim:
            hidden_dim:
            output_dim:
        """
        super(ReflectMLP, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.discretized = discretized

        self.relu = nn.ReLU()

        if not self.discretized:
            self.bn = nn.BatchNorm1d(hidden_dim)

        ' Head 1: Classification (y) '
        if self.discretized:
            self.y_head = nn.Linear(hidden_dim, output_dim*3)
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.y_head = nn.Linear(hidden_dim, output_dim)

        ' Head 2: REINFORCE (r): Logits for binary actions '
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.r_head = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)

        if not self.discretized:
            emb = self.bn(emb)

        ' clf head '
        output_y = self.y_head(emb)
        output_y = self.softmax(output_y.view(output_y.shape[0], -1, 3))\
                if self.discretized else output_y

        ' action head '
        #output_r = self.r_head(self.relu(self.fc(emb)))
        output_r = self.r_head(emb)
        output_r = self.sigmoid(output_r)

        return output_y, output_r

    def predict(self,x):
        output_y, _ = self.forward(x)
        return torch.argmax(output_y, dim=-1) -1\
                 if self.discretized else torch.where(torch.abs(output_y)>.1, torch.sign(output_y), 0)

    def reflection(self, x):
        _, output_r = self.forward(x)
        return torch.round(output_r.detach())

################################################################################

class ReflectGNN(nn.Module):
    """ Network Structure of Base Learner with Reflect Output (RL) """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layers,
                 output_dim,
                 gnn_extra_layer=False,
                 device='cpu',
                 label_mask=None,
                 discretized=True):
        '''
        Network Struct of GNN
        Args:
            input_dim: 
            hidden_dim:
            num_layers:
            output_dim:
            device:
            label_mask:
        '''
        super(ReflectGNN, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.discretized = discretized

        self.input_emb = nn.Embedding(self.input_dim, hidden_dim, max_norm=True)
        
        'edge idx & weight as buffers'
        self.register_buffer('edge_index', None)
        self.register_buffer('edge_weight', None)

        'GNN layers'
        self.graph_layers = torch.nn.ModuleList()
        for _ in range(1, self.num_layers + 1):
            self.graph_layers.append(SGConv(hidden_dim, hidden_dim, 1))
    
        self.bn= nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        ' Head 1: Classification (y) '
        self.y_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, 3 * self.output_dim)\
                if self.discretized else\
                nn.Linear(hidden_dim*2, self.output_dim),
        ) if not gnn_extra_layer else nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3 * self.output_dim)\
                if self.discretized else\
                nn.Linear(hidden_dim, self.output_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

        ' Head 2: REINFORCE (r): Logits for binary actions '
        self.r_head = nn.Linear(hidden_dim, output_dim)\
                if not gnn_extra_layer else nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, output_dim),
                )
        self.sigmoid = nn.Sigmoid()

    def set_weighted_adjacency(self, adj_matrix):
        """
        Set the weighted adjacency matrix for the fixed graph.
        Args:
            adj_matrix: [input_dim, input_dim] weighted adjacency matrix
        """
        self.edge_index, self.edge_weight = dense_to_sparse(adj_matrix)
        
        row, col = self.edge_index
        deg = torch.sparse_coo_tensor(
            torch.stack([row, row]), 
            self.edge_weight, 
            (self.input_dim, self.input_dim)
        ).to_dense().sum(1)
        deg_inv_sqrt = torch.abs(deg).pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #deg_inv_sqrt *= torch.sign(deg)
        norm = deg_inv_sqrt[row] * self.edge_weight * deg_inv_sqrt[col]

        self.edge_weight = torch.abs(norm)

    def forward(self, x):
        """
        Args:
            x: One-hot encoded node features [batch_size, input_dim, input_dim]
                     or [input_dim, input_dim] if single graph
        Returns:
            logits: Classification logits [batch_size, output_dim, 3] or [output_dim, 3]
        """
        if self.edge_index is None:
            raise RuntimeError("Adjacency matrix not set. Call set_weighted_adjacency() first.")
        if len(x.shape) == 2:
            batch_size = x.shape[0]
        elif len(x.shape) == 1:
            batch_size = 1
            x = x.unsqueeze(0)
        else:
            raise RuntimeError("Input dimension error")

        ' init embeddings '
        emb = self.input_emb(torch.LongTensor(list(range(self.input_dim))).to(self.device))
        emb = self.relu(self.bn(emb))

        ' apply GNN layers '
        for i, layer in enumerate(self.graph_layers):
            emb = layer(emb, self.edge_index, self.edge_weight)
            if i < self.num_layers - 1:
                emb = self.relu(emb)

        ' add GNN embedding to corresponding input '
        emb = x @ emb

        ' clf head '
        output_y = self.y_head(emb)
        output_y = self.softmax(output_y.view(output_y.shape[0], -1, 3))\
                if self.discretized else self.relu(output_y)

        ' action head '
        output_r = self.r_head(emb)
        output_r = self.sigmoid(output_r)

        return output_y, output_r

    def predict(self,x):
        output_y, _ = self.forward(x)
        return torch.argmax(output_y, dim=-1) -1\
                if self.discretized else  torch.where(torch.abs(output_y)>.1, torch.sign(output_y), 0)

    def reflection(self, x):
        _, output_r = self.forward(x)
        return torch.round(output_r.detach())

################################################################################
################################################################################

class ReflectLearner():
    def __init__(self,
        input_dim,
        output_dim,
        hidden_dim = 64,
        base_learner_type = 'MLP',
        num_layers = 3,
        adj_matrix = None | torch.Tensor,
        discretized = True,
        gnn_extra_layer = False,
        device = 'cpu',
        log_path = '',
    ) -> None:
        '''
        Args:
            input_dim:
            output_dim:
            hidden_dim:
            base_learner_type:
            num_layers:
            adj_matrix:
            device:
            log_path (optional):
        '''
        self.device = device
        if device != 'cpu':
            print(f'cuda availability: {torch.cuda.is_available()}')
            assert torch.cuda.is_available()

        self.discretized = discretized

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ' weight of classes for CE loss '
        self.clf_weight = torch.Tensor([.4,.2,.4])

        if base_learner_type == 'MLP':
            self.model = ReflectMLP(self.input_dim,
                                    self.hidden_dim,
                                    self.output_dim,
                                    discretized=self.discretized)
        elif base_learner_type == 'GNN':
            assert adj_matrix != None
            self.model = ReflectGNN(self.input_dim,
                                    self.hidden_dim,
                                    num_layers,
                                    self.output_dim,
                                    gnn_extra_layer,
                                    self.device,
                                    discretized=self.discretized)
            self.model.set_weighted_adjacency(adj_matrix)
        else:
            raise Exception('Invalid Base Learner Type')

        self.train_loader = None
        self.test_loader = None

        self.model = self.model.to(self.device)
        self.clf_weight = self.clf_weight.to(self.device)

        self.log_path = log_path


    def consistency_reward(self,
                           KB: RegulatoryKB,
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

        y = torch.argmax(y_probs, dim=-1) -1\
                if self.discretized else torch.where(torch.abs(y_probs)>.1, torch.sign(y_probs), 0)

        violated = KB.violated(Y=y, X=x, mask=~(r_binary.bool()))
        #violated = KB.violated(Y=torch.where(r_binary.bool(), KB.deduce(x), y), X=x)

        weighted_restriction = torch.sum(torch.clamp(
            torch.sign(r_binary - .5) * (.5 - label_weight), min=0))\
                    if label_weight != None else 0
        #weighted_restriction = torch.sum(r_binary @ (1-label_weight) + (1-r_binary) @ label_weight)

        #total = r_binary.shape[0] * r_binary.shape[1]
        #len_restriction = torch.max(torch.count_nonzero(r_binary) - th * total, other=torch.tensor(0))

        return - .1*violated -  weighted_restriction #- len_restriction
        #return - weighted_restriction

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

    def init_weight(self, label_weight: torch.Tensor, epochs=1000, lr=1e-4):
        #"""
        #Initialize a linear layer to produce desired outputs after sigmoid
        
        #Args:
            #label_weight: torch.Tensor - desired initial output values (0 <= y <= 1)
        #"""
        #with torch.no_grad():
            ## Clamp to avoid numerical instability
            #y = torch.clamp(label_weight, 1e-7, 1-1e-7)
            
            ## Compute required biases (logits)
            #bias_data = torch.log(y / (1 - y))
            
            ## Set biases
            #self.model.r_head.bias.data = bias_data
            
            ## Set weights to small random values
            #nn.init.normal_(self.model.r_head.weight, mean=0, std=0.01)

        """
        Train only the last layer to produce desired outputs

        Args:
            model: nn.Module with sigmoid output
            desired_output: torch.Tensor of shape (batch_size, output_dim)
            input_samples: torch.Tensor of representative input samples
            epochs: training iterations
            lr: learning rate
        """
        # Freeze all layers except last
        for name, param in self.model.named_parameters():
            #if not name.startswith('r_head' if hasattr(self.model, 'r_head')
            #                else name.startswith('net.' + str(len(self.model.net)-1))):
            if not name.startswith('r_head'):
                param.requires_grad = False

        # Set up optimization
        criterion = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        # Training loop
        input_samples =  torch.eye(self.input_dim).to(self.device)# if X == None else\
                #torch.concat([X, torch.eye(self.input_dim).to(self.device)])
        desired_output = label_weight.unsqueeze(0).expand(input_samples.shape[0], -1)
        for epoch in range(epochs):
            optimizer.zero_grad()
            _,outputs = self.model(input_samples)
            loss = criterion(outputs, label_weight)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            if loss.item() < 1e-4:  # Early stopping
                break

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        print("Final loss:", loss.item())
        print("Achieved outputs:", self.model(input_samples)[1].detach())



    def train(
        self, 
        KB: RegulatoryKB,
        label_weight= None | torch.Tensor,
        epochs= 10, 
        reinforce_epochs= 100,
        C= 1,
        lr= 1e-3, 
        lr_decay= 1.,
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
        if self.discretized:
            criterion = nn.CrossEntropyLoss(weight=self.clf_weight)
        else:
            criterion = nn.MSELoss(reduction='mean')

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if lr_decay < 1.:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
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
                if self.discretized:
                    Y_batch = Y_batch.to(int)+1

                ' CE loss '
                loss_y += criterion(output_y.view(-1,3), Y_batch.view(-1)) if self.discretized\
                        else criterion(output_y, Y_batch)

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
            if lr_decay < 1.:
                scheduler.step()
            total_loss.backward()
            optimizer.step()

            if (epoch+1)%2000== 0: #NOTE tmp
                self.eval(KB, w_data=.3, write_log=False, verbose=True)

            if (epoch+1)%100 == 0 and verbose:
                print(f"Epoch {epoch+1}, Total loss: {total_loss.item():.4f}, CE loss: {loss_y.item():.4f}, RL loss: {loss_r.item():.4f}, Reward: {reward:.4f}")



                #NOTE TMP #############################
                for X_batch, _ in self.train_loader:
                    output_y, output_r = self.model(X_batch)
                    y = torch.argmax(output_y, dim=-1) -1 if self.discretized\
                            else torch.where(torch.abs(output_y)>.1, torch.sign(output_y), 0)
                    r = torch.round(output_r)
                    print(f'    full cols: {torch.count_nonzero(torch.sum(r,dim=0)==len(r))}, non-full cols: {torch.count_nonzero((torch.sum(r,dim=0)<len(r)) & (torch.sum(r,dim=0)>0))}')
                    #print(f'r={output_r}')
                    
                    labels = torch.nonzero(label_weight > .5).squeeze(-1).cpu().detach().numpy().tolist()
                    r_idx = torch.nonzero(torch.sum(r,dim=0)).squeeze(-1).cpu().detach().numpy().tolist()
                    print(f'   r - labels: {len(set(r_idx)-set(labels))}, labels - r: {len(set(labels) - set(r_idx))}')
                    #print(f'    r-labels: {output_r[0,list(set(r_idx)-set(labels))]}\n    labels-r: {output_r[0,list(set(labels)-set(r_idx))]}')
                    
                    violated = KB.violated(Y=y, X=X_batch, mask=~(r.bool()))
                    #violated = KB.violated(Y=torch.where(r.bool(), KB.deduce(X_batch), y), X=X_batch)
                    weighted_restriction = torch.sum(torch.clamp(
                        torch.sign(r- .5) * (.5-label_weight), min=0))\
                                if label_weight != None else 0
                    #weighted_restriction = torch.sum(r @ (1-label_weight) + (1-r) @ label_weight)
                    total = r.shape[0] * r.shape[1]
                    len_restriction = torch.max(torch.count_nonzero(r) - .3 * total, other=torch.tensor(0))
                    print(f'    violated: {violated}, weighted: {weighted_restriction}, len: {len_restriction}, nonzero: {torch.count_nonzero(r) / total}')
                    break
                # NOTE TMP ############################


    #########################################################################

    def eval(self,
             KB: RegulatoryKB,
             w_data = None | float,
             p_integrate = 2,
             write_log=True,
             verbose=False):

        assert self.test_loader != None
        w_data = .5 if w_data == None or w_data<0. or w_data>1. else w_data

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():

            Y_test, Y_pred, Y_prob, Y_deduc = [],[],[],[]
            R_pred = []

            for X_batch, Y_batch in self.test_loader:
                outputs = self.model.predict(X_batch)
                r_pred = self.model.reflection(X_batch)

                Y_test.append(Y_batch)
                Y_pred.append(outputs)
                R_pred.append(r_pred)
            #    Y_prob.append(self.predict_prob(X_batch).max(dim=-1).values)
                total += Y_batch.size(0)
                correct += (outputs == Y_batch).sum(dim=0)
                Y_deduc.append(KB.deduce(X_batch))

            Y_test = torch.concat(Y_test, dim=0)
            Y_pred = torch.concat(Y_pred, dim=0)
            R_pred = torch.concat(R_pred, dim=0)

            Y_deduc = torch.concat(Y_deduc, dim=0)
            Y_refl = torch.where(R_pred.bool(), Y_deduc, Y_pred)

            if self.device != 'cpu':
                Y_test = Y_test.cpu()
                Y_pred = Y_pred.cpu()
                Y_refl = Y_refl.cpu()
                Y_deduc = Y_deduc.cpu()
                correct = correct.cpu()

            #Y_prob = torch.concat(Y_prob, dim=0)

            ''' flatten '''
            flat_y_t = Y_test.flatten()
            flat_y_p = Y_pred.flatten()
            flat_y_r = Y_refl.flatten()
            flat_y_d = Y_deduc.flatten()


            ' performance of prediction result '
            confusion_pred = confusion_matrix(flat_y_t, flat_y_p, labels=[-1, 0,1])
            confusion_pred = confusion_pred / np.sum(confusion_pred)
            f1_pred_macro = f1_score(flat_y_t, flat_y_p, average='macro') # micro on labels, macro on classes
            f1_pred_micro = f1_score(flat_y_t, flat_y_p, average='micro') # micro on labels, micro on classes
            f1_pred_kb = f1_score(flat_y_d, flat_y_p, average='macro')
            f1_pred_final = (w_data * (f1_pred_macro ** -p_integrate)\
                    + (1.-w_data) * (f1_pred_kb ** -p_integrate)) ** (-1/p_integrate)

            ' performance of integrated result '
            confusion_refl = confusion_matrix(flat_y_t, flat_y_p, labels=[-1, 0,1])
            confusion_refl = confusion_refl / np.sum(confusion_pred)
            f1_refl_macro = f1_score(flat_y_t, flat_y_r, average='macro') # micro on labels, macro on classes
            f1_refl_micro = f1_score(flat_y_t, flat_y_r, average='micro') # micro on labels, micro on classes
            f1_refl_kb = f1_score(flat_y_d, flat_y_r, average='macro')
            f1_refl_final = w_data * f1_refl_macro + (1.-w_data) * f1_refl_kb


            ''' compute acc & confusion matrix on each gene '''
            per_label_accuracy = correct / total
            #f1 /= Y_test.shape[1]
            if self.log_path != '' and write_log:
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

                    f.write(f'\n\n------\nprediction result:\nconfusion matrix:\n{confusion_pred}\n')
                    f.write(f'f1 on test:    {f1_pred_macro}\n')
                    f.write(f'f1 on test:    {f1_pred_micro} (micro)\n')
                    f.write(f'f1 on KB:      {f1_pred_kb}\n')
                    f.write(f'integrated f1: {f1_pred_final}, w_data: {w_data}, w_klg: {1-w_data}\n\n')

                    f.write(f'\n\n------\nintegrated result:\nconfusion matrix:\n{confusion_refl}\n')
                    f.write(f'f1 on test:    {f1_refl_macro}\n')
                    f.write(f'f1 on test:    {f1_refl_micro} (micro)\n')
                    f.write(f'f1 on KB:      {f1_refl_kb}\n')
                    f.write(f'integrated f1: {f1_refl_final}, w_data: {w_data}, w_klg: {1-w_data}\n\n')
                    #f.write(f'weighted f1: {f1_weighted}\n')
                    #f.write(f'class -1 f1: {f1_class[0]}\n')
                    #f.write(f'class  0 f1: {f1_class[1]}\n')
                    #f.write(f'class  1 f1: {f1_class[2]}\n')
                    #f.write(f'average label-wise acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')


            if verbose:
                print(f'--- eval ---\nprediction result:')
                print(f'f1 on test:    {f1_pred_macro:.4f}')
                print(f'f1 on test:    {f1_pred_micro:.4f} (micro)')
                print(f'f1 on KB:      {f1_pred_kb:.4f}')
                print(f'integrated f1: {f1_pred_final:.4f}, w_data: {w_data:.4f}, w_klg: {1-w_data:.4f}')

                print(f'\nintegrated result:')
                print(f'f1 on test:    {f1_refl_macro:.4f}')
                print(f'f1 on test:    {f1_refl_micro:.4f} (micro)')
                print(f'f1 on KB:      {f1_refl_kb:.4f}')
                print(f'integrated f1: {f1_refl_final:.4f}, w_data: {w_data:.4f}, w_klg: {1-w_data:.4f}')
                print('------------')

            return f1_pred_final

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def predict(self, x: torch.Tensor):
        return self.model.predict(x)

    def reflection(self, x: torch.Tensor):
        return self.model.reflection(x)

    def predict_prob(self, x: torch.Tensor):
        outputs, _ = self.model(x)
        return outputs

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)



if __name__ == '__main__':
    # NOTE tmp test

    torch.manual_seed(0)
    data_name = 'norman'
    np.random.seed(0)
    device = 'cuda:3'
    log_file = 'log/learner.txt'

    X_train = torch.tensor(load_npz(f'dataset/human/{data_name}_X.npz').toarray(), dtype = torch.float32)
    Y_train = torch.tensor(load_npz(f'dataset/human/{data_name}_Y_con.npz').toarray(), dtype = torch.float32)
    label_weight = torch.tensor(np.load(f'dataset/human/{data_name}_label_weight.npy'))
    
    p_train = 1.
    test_idx = np.zeros(shape=len(X_train), dtype=bool)
    test_idx[np.load(f'dataset/human/{data_name}_test_idx.npy')] = True

    train_idx = np.random.choice([True, False], size=len(X_train)-np.count_nonzero(test_idx), p=[p_train, 1-p_train])

    X_test = X_train[test_idx]
    Y_test = Y_train[test_idx]
    X_train = X_train[~ test_idx][train_idx]
    Y_train = Y_train[~ test_idx][train_idx]
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    label_weight = label_weight.to(device)

    reasoner = RegulatoryKB(pos_trn_pth= f'rules/human/{data_name}_GO.npz',
                            neg_trn_pth= None,
                            output_idx_list= None,
                            device=device)
    reasoner.closure_(T=5, closure_type='naive')

    adj_matrix = torch.round(torch.clamp(torch.abs(reasoner.Regu_P_0 + reasoner.Regu_N_0), 0,1))
    learner = ReflectLearner(input_dim= X_test.shape[1],
                             output_dim= Y_test.shape[1],
                             hidden_dim= 64,
                             base_learner_type= 'MLP',
                             adj_matrix= adj_matrix,
                             device=device,
                             discretized=False,
                             gnn_extra_layer=True,
                             log_path=log_file)

    criterion = nn.MSELoss(reduction='mean')
    Y_pred, _ = learner.forward(X_test)
    print(f'MSE baseline: {criterion(torch.zeros_like(Y_test).to(device), Y_test.detach())}')
    print(f'MSE before training: {criterion(Y_pred.detach(), Y_test.detach())}')

    learner.load_data(X_train, Y_train, X_test, Y_test)
    learner.train(KB= reasoner,
                  label_weight= label_weight,
                  epochs= 1000,
                  reinforce_epochs= 1,
                  C=1,
                  lr=1e-3,
                  lr_decay=.999,
                  verbose=True)

    Y_pred, _ = learner.forward(X_test)
    print(f'MSE: {criterion(Y_pred.detach(), Y_test.detach())}')

    Y_test = torch.tensor(load_npz(f'dataset/human/{data_name}_Y.npz').toarray(), dtype = int)[test_idx].to(device)
    learner.load_data(_, _, X_test, Y_test)
    f1 = learner.eval(reasoner, .3, verbose=True)
    print(f'pretrain: integrated f1 {f1:.4f}')
    

    #Y_train = torch.tensor(np.load('dataset/precise1k/Y_label.npy'), dtype=int)
    #X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype=torch.float32)
    #Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_label.npy'), dtype=int)

    #import pandas as pd
    #label_set = pd.read_csv('dataset/label_set_iml.csv')
    #idx_list_p1k = list(label_set['precise1k_idx'])
    #idx_list_sra = list(label_set['matrix_idx'])

    #Y_train = Y_train[:,idx_list_p1k]
    #Y_test = Y_test[:,idx_list_sra]

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #X_train, Y_train = X_train.to(device), Y_train.to(device)
    #X_test, Y_test = X_test.to(device), Y_test.to(device)



    #input_dim = X_train.shape[1]
    #output_dim = Y_train.shape[1]
    #hidden_dim = 128
    #batch_size = 64
    #

    ## Initialize model
    ##data_loader = DataLoader(TensorDataset(X_train,Y_train), batch_size=batch_size, shuffle=True)
    ##learner.train_loader = data_loader

    ## Train
    #learner = ReflectLearner(input_dim=X_train.shape[1], output_dim=Y_train.shape[1], device=device, log_path='log.txt')
    #regulatory_kb = RegulatoryKB(pos_trn_pth= 'rules/regu_pos.npz', neg_trn_pth='rules/regu_neg.npz', output_idx_list=idx_list_sra, device=device)
    #regulatory_kb.closure_(T=5, closure_type='weighted')

    #learner.load_data(X_train, Y_train, X_test, Y_test, batch_size=batch_size)
    #print(learner.eval())
    #learner.train(KB= regulatory_kb, epochs=50000, lr=1e-4)
    #print(learner.eval())
