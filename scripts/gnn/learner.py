import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from scipy.sparse import load_npz
import pandas as pd

import torch.nn.functional as F
from torch_geometric.data import batch
from torch_geometric.nn import SGConv
from torch_geometric.utils import dense_to_sparse
from torch.nn.utils.rnn import pad_sequence

import cupy as cp
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix

def closure(R0_P, R0_N, T=None, print_matrix=False):

    ''' get dim of matrix KB
        note. positive & negative regultion KB should of same size '''
    dim = R0_P.shape[0]
    assert dim==R0_N.shape[0]

    R0_P = cp.asarray(R0_P, dtype=cp.bool_)
    R0_N = cp.asarray(R0_N, dtype=cp.bool_)

    ''' R*_P = sum ((R_P)^i R_P + (R_N)^i R_N)
        R*_N = sum ((R_P)^i R_N + (R_N)^i R_P) '''
    I = cp.eye(dim, dtype=cp.bool_)
    R_P = I + R0_P
    R_N = I + R0_N
    #R_N = identity(dim) + N

    if print_matrix:
        print(f'initial R_P = R_P + I = \n{str(R_P)}\ninitial R_N =\n{str(R_N)}\n----------\n')

    cnt = 0
    while True:
        ' multiply until closure '
        R_P_ = R_P @ (R0_P+I) + R_N @ (R0_N+I)
        R_N_ = R_P @ (R0_N+I) + R_N @ (R0_P+I)
        if print_matrix:
            print(f'{cnt}th iteration:\nR{cnt}_P =\n{str(R_P_)}\nR{cnt}_N =\n{str(R_N_)}\n\n')
        if cp.all(R_P_ == R_P) and cp.all(R_N_ == R_N):
            break
        if T and cnt>=T:
            break
        cnt += 1
        R_P = R_P_
        R_N = R_N_

    res_P = R_P @ R0_P
    res_N = R_N @ R0_N
    if print_matrix:
        print(f'\n----------\nfinal result:\nR*_P =\n{str(res_P)}\nR*_N =\n{str(res_N)}\n')
    return res_P, res_N, cnt

class PertDataset(Dataset):
    ' define dataset for GNN pert data '
    def __init__(self, X, Y):
        assert len(X) == len(Y), "Input and output must have same length"
        self.X, self.Y = X,Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def collate_fn(batch):
    """Custom collate function to handle variable-length inputs"""
    inputs = [item[0] for item in batch]
    outputs = torch.stack([item[1] for item in batch])

    # Pad variable-length input sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_inputs, outputs

class GraphNN(nn.Module):
    ''' network struct of base learner '''

    def __init__(self, num_genes, hidden_dim, num_layers, output_dim, device='cpu', label_mask=None):
        '''
        Network Struct of GNN
        Args:
            num_genes: 
            hidden_dim:
            num_layers:
            output_dim:
            device:
            label_mask:
        '''
        super(GraphNN, self).__init__()
        self.device = device

        self.num_genes = num_genes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_emb = nn.Embedding(self.num_genes, hidden_dim, max_norm=True)
        
        'edge idx & weight as buffers'
        self.register_buffer('edge_index', None)
        self.register_buffer('edge_weight', None)

        'GNN layers'
        self.graph_layers = torch.nn.ModuleList()
        for _ in range(1, self.num_layers + 1):
            self.graph_layers.append(SGConv(hidden_dim, hidden_dim, 1))

        '3-clf head for prediction output'
        self.fc = nn.Linear(hidden_dim, hidden_dim*2)
        self.classifier = nn.Linear(hidden_dim*2, 3 * self.output_dim)
        
    
        self.bn= nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

        self.label_mask = label_mask

    def set_weighted_adjacency(self, adj_matrix):
        """
        Set the weighted adjacency matrix for the fixed graph.
        Args:
            adj_matrix: [num_genes, num_genes] weighted adjacency matrix
        """
        self.edge_index, self.edge_weight = dense_to_sparse(adj_matrix)
        
        row, col = self.edge_index
        deg = torch.sparse_coo_tensor(
            torch.stack([row, row]), 
            self.edge_weight, 
            (self.num_genes, self.num_genes)
        ).to_dense().sum(1)
        deg_inv_sqrt = torch.abs(deg).pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #deg_inv_sqrt *= torch.sign(deg)
        norm = deg_inv_sqrt[row] * self.edge_weight * deg_inv_sqrt[col]

        self.edge_weight = torch.abs(norm) #NOTE no negative weights
    

    def forward(self, x, use_gpu=False):
        """
        Args:
            x_onehot: One-hot encoded node features [batch_size, num_genes, num_genes]
                     or [num_genes, num_genes] if single graph
        Returns:
            logits: Classification logits [batch_size, num_genes, 3] or [num_gnees, 3]
        """
        if self.edge_index is None:
            raise RuntimeError("Adjacency matrix not set. Call set_weighted_adjacency() first.")
        if len(x.shape) == 3:
            batch_size = x.shape[0]
        elif len(x.shape) == 2:
            batch_size = 1
            x = x.unsqueeze(0)
        else:
            raise RuntimeError("Input dimension error")

        ' init embeddings '
        if use_gpu:
            emb = self.input_emb(torch.LongTensor(list(range(self.num_genes))).to(self.device))
        else:
            emb = self.input_emb(torch.LongTensor(list(range(self.num_genes))))
        emb = self.relu(self.bn(emb))

        ' apply GNN layers '
        for i, layer in enumerate(self.graph_layers):
            emb = layer(emb, self.edge_index, self.edge_weight)
            if i < self.num_layers - 1:
                emb = self.relu(emb)

        ' add GNN embedding to corresponding input '
        out = torch.zeros((batch_size, self.hidden_dim))
        if use_gpu:
            out = out.to(self.device)
        for i in range(batch_size):
            for item in x[i]:
                out[i] = out[i] + emb[item[0]] * item[1]

        ' 3-clf head '
        out = self.relu(out)
        out = self.relu(self.fc(out))
        out = self.classifier(out)
        out = out.view(batch_size, -1, 3)
        out = self.softmax(out)

        return out

    def predict(self,x):
        output = self.forward(x)
        output = torch.argmax(output, dim=-1) -1
        if self.label_mask != None:
            output = output[self.label_mask] if output.dim==1 else output[:,self.label_mask]
        return output

class GnnLearner():
    def __init__(self, adj_matrix, label_mask = None, log_path = '') -> None:

        ' 4639 genes of whole genome '
        num_genes = 4639

        ' GNN hidden dim & layers '
        hidden_dim = 64
        num_layers = 5

        ' 241 output genes '
        output_dim = 623

        ' weight of classes for CE loss '
        self.clf_weight = torch.Tensor([.4,.2,.4])

        self.label_mask = torch.tensor(label_mask) if label_mask!=None else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GraphNN(num_genes, hidden_dim, num_layers, output_dim, device=self.device)
        self.train_loader = None
        self.test_loader = None
        self.adj_matrix = adj_matrix
        self.model.set_weighted_adjacency(self.adj_matrix)


        print(torch.cuda.is_available())

        #if log_path != '':
        #    with open(log_path, 'w') as f:
        #        f.write('')
        self.log_path = log_path

    def load_data(self, X_train, Y_train, X_test, Y_test, batch_size=64):
        ''' define train & test data loader '''
        #assert len(X_train) > 0
        #assert len(X_test) > 0
        assert len(Y_train) > 0
        assert len(Y_test) > 0

        #NOTE tmp
        import pandas as pd
        gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
        gene_idx['index'] = gene_idx.index
        train_metadata = pd.read_csv('dataset/precise1k/metadata.csv', index_col=0)
        gene_idx_locus = gene_idx.set_index('locus')
        X_train = [torch.tensor([[int(gene_idx_locus.loc[k,'index']),int(v)] for k,v in eval(d).items()]) for d in train_metadata['perturbation']]
        X_test_lst = [[] for _ in range(len(X_test))]
        for item in torch.nonzero(X_test):
            X_test_lst[item[0]].append([item[1],1])
        X_test = [torch.tensor(x) if len(x)>0 else torch.tensor([[0,0]]) for x in X_test_lst]



        print(len(X_train), len(Y_train))
        print(len(X_test), len(Y_test))
        train_dataset = PertDataset(X_train, Y_train)
        test_dataset = PertDataset(X_test, Y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        ''' reset classification loss weight with new Y_train '''
        flat_y = Y_train.flatten()
        weights = [1/(torch.sum(flat_y==-1).item() + 1e-6),
                   1/(torch.sum(flat_y==0).item() + 1e-6),
                   1/(torch.sum(flat_y==1).item() + 1e-6)]
        self.clf_weight = torch.Tensor(weights) / sum(weights)


    def train(self, epochs: int, mask=None, use_gpu=False, lr=0.001):
        assert self.train_loader != None
        #if self.log_path != '':
        #    with open(self.log_path,'a') as f:
        #        f.write('---------- Train ----------\n')

        ' load model to gpu '
        if use_gpu:
            self.model = self.model.to(self.device)
            self.clf_weight = self.clf_weight.to(self.device)

        if self.label_mask != None:
            if use_gpu:
                self.label_mask = self.label_mask.to(self.device)
            if self.train_loader.batch_size > 1:
                label_mask = self.label_mask[None, ..., None]
            else:
                label_mask = self.label_mask[..., None]

        ''' Training loop '''
        criterion = nn.CrossEntropyLoss(weight=self.clf_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for X_batch, Y_batch in self.train_loader:
                if use_gpu:
                    X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                    batch_size = len(X_batch)

                ' Forward pass '
                outputs = self.model(X_batch, use_gpu)

                if self.label_mask != None:
                    outputs = torch.masked_select(outputs, label_mask)

                if mask:
                    if use_gpu:
                        mask = mask.to(self.device)
                    selected_outputs = torch.masked_select(outputs, mask)
                    selected_targets = torch.masked_select((Y_batch+1), mask)
                    loss = criterion(selected_outputs.view(-1,3), selected_targets.view(-1))
                else:
                    loss = criterion(outputs.view(-1,3), (Y_batch+1).view(-1))
                
                ''' Backward pass and optimization '''
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()


            if self.log_path != '':# and (epoch+1) % 10 == 0:
                with open(self.log_path,'a') as f:
                    f.write(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.train_loader):.4f}\n")
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.train_loader):.4f}\n")

        ' move model back to cpu '
        if use_gpu:
            self.model = self.model.to("cpu")
        
    def eval(self):
        assert self.test_loader != None
        #if self.log_path != '':
        #    with open(self.log_path,'a') as f:
        #        f.write('---------- Eval ----------\n')

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            f1_micro2 = 0
            f1_macro2 = 0

            Y_test, Y_pred, Y_prob = [],[],[]

            for X_batch, Y_batch in self.test_loader:
                outputs = self.model.predict(X_batch)
                print(outputs.shape)

                Y_test.append(Y_batch)
                Y_pred.append(outputs)
                Y_prob.append(self.predict_prob(X_batch).max(dim=-1).values)
                total += Y_batch.size(0)
                correct += (outputs == Y_batch).sum(dim=0)

            Y_test = torch.concat(Y_test, dim=0)
            Y_pred = torch.concat(Y_pred, dim=0)
            Y_prob = torch.concat(Y_prob, dim=0)

            ''' compute total confusion matrix '''
            flat_y_t = Y_test.flatten()
            flat_y_p = Y_pred.flatten()
            print(flat_y_p.shape, flat_y_t.shape)
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
                    #f.write('\n---- data ----')

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
                    #f.write(f'macro f1 2: {f1_macro2}\n')
                    #f.write(f'micro f1 2: {f1_micro2}\n')
                    f.write(f'weighted f1: {f1_weighted}\n')
                    f.write(f'class -1 f1: {f1_class[0]}\n')
                    f.write(f'class  0 f1: {f1_class[1]}\n')
                    f.write(f'class  1 f1: {f1_class[2]}\n')
                    f.write(f'average label-wise acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')
            else:
                print(f'Average Per-label Acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')

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
    torch.cuda.set_per_process_memory_fraction(1., device=0) 



    X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype=torch.float32)
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_train.npy'), dtype=int)
    test_idx = torch.tensor(np.random.choice([True, False], size=len(X_train), p=[.2, .8]))
    X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype=torch.float32)
    Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_train.npy'), dtype=int)
    # NOTE Temp
    #X_test, Y_test = X_train[test_idx], Y_train[test_idx]
    #X_train, Y_train = X_train[~test_idx], Y_train[~test_idx]

    label_set = pd.read_csv('dataset/ncbi-sra/label_set.csv', index_col=0)
    label_mask = list(label_set['matrix_idx'])

    pos_regu = load_npz('rules/regu_pos.npz').toarray()#[:,label_mask]
    neg_regu = load_npz('rules/regu_neg.npz').toarray()#[:,label_mask]
    pos_regu, neg_regu,_ = closure(pos_regu, neg_regu)
    adj_matrix = np.zeros_like(pos_regu, dtype=np.int8)
    adj_matrix[:, :] = np.where(
            (pos_regu[:,:]!=0) & (neg_regu[:,:]!=0) , 0.5,
        np.where((pos_regu[:,:]!=0) , 1, 
                 np.where((neg_regu[:,:]!=0) , -1, 0.1)))
    adj_matrix = torch.tensor(adj_matrix)


    ''' add logger & training '''
    from datetime import datetime
    log_file = f'log/GNN-{datetime.now()}.txt'.replace(' ','-')
    label_mask = [True if i in label_mask else False for i in range(X_train.shape[1])]
    learner = GnnLearner(adj_matrix=adj_matrix,  log_path=log_file)

    learner.load_data(X_train, Y_train, X_test, Y_test)
    learner.eval()
    learner.train(epochs=50, use_gpu=True)
    learner.eval()
