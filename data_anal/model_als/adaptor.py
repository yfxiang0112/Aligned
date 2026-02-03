import torch
import numpy as np
from aligned.reasoner import RegulatoryKB
from aligned.learner_adap import AdaptorLearner
from scipy.sparse import load_npz
import json
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

data_name = 'norman'
device = 'cuda:6'

Y = torch.tensor(load_npz(f'dataset/human/{data_name}_Y.npz').toarray()).to(device).float()
X = torch.tensor(load_npz(f'dataset/human/{data_name}_X.npz').toarray()).to(device).float()
total = len(X)

reasoner = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device=device)
reasoner.closure_(T=5, closure_type='weighted')

adj_matrix = torch.round(torch.clamp(torch.abs(reasoner.Regu_N_0 + reasoner.Regu_P_0), 0,1))
learner = AdaptorLearner(input_dim= X.shape[1],
                         output_dim= Y.shape[1],
                         hidden_dim= 64,
                         base_learner_type= 'GNN',
                         adj_matrix= adj_matrix,
                         device=device)

label_weight = torch.tensor(np.load(f'dataset/human/{data_name}_label_weight.npy')).to(device)
print('w>.5: ', torch.count_nonzero(label_weight>=.5).item())

for repl in range(5):
    reasoner.load(f'data_anal/experiment_results/{data_name}/models/GNN_abl0_{repl+1}.npz')
    learner.load(f'data_anal/experiment_results/{data_name}/models/GNN_abl0_{repl+1}.pt')
    
    Y_p = learner.predict(torch.tensor(X).float().to(device)).float()
    R = learner.reflection(torch.tensor(X).float().to(device))
    Y_d= reasoner.deduce(torch.tensor(X).float().to(device)).float()
    Y_r = torch.where(R>=.5, Y_d, Y_p).float()

    R_col = torch.sum(R, dim=0)/len(R)
    print('R=1        > 0:  ', torch.count_nonzero( R_col).item())
    print('R=1 & w>.5 > 0:  ', torch.count_nonzero((R_col > 0) & (label_weight>=.5)).item())
    print('R=1        > 10%:', torch.count_nonzero( R_col > .1).item())
    print('R=1 & w>.5 > 10%:', torch.count_nonzero((R_col > .1) & (label_weight>=.5)).item())
    print('R=1        > 90%:', torch.count_nonzero( R_col > .9).item())
    print('R=1 & w>.5 > 90%:', torch.count_nonzero((R_col > .9) & (label_weight>=.5)).item())

    direct = (torch.sum(torch.abs(reasoner.Regu_P_0)\
            +torch.abs(reasoner.Regu_N_0), dim=0))# / len(reasoner.Regu_P_0)
    direct = torch.clamp(direct, 0,1)
    indirect = (torch.sum(torch.abs(reasoner.KB_P - reasoner.Regu_P_0)\
            +torch.abs(reasoner.KB_N - reasoner.Regu_N_0), dim=0))# / len(reasoner.KB)
    indirect = torch.clamp(torch.clamp(indirect, 0,1) - direct, 0,1)

    #corr, p_value = pearsonr(R_col.cpu().numpy(), direct.cpu().numpy())
    #print('corr direct:  ',corr)
    #corr, p_value = pearsonr(R_col.cpu().numpy(), indirect.cpu().numpy())
    #print('corr indirect:',corr)
    R_col = torch.round(R_col)
    print(torch.sum((R_col==1)&(direct==1))/ torch.sum(direct==1))
    print(torch.sum((R_col==1)&(indirect==1))/ torch.sum(indirect==1))

    exit()
    del Y_p
    del Y_r
    del R
    del Y_d
    torch.cuda.empty_cache()
