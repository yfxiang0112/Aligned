import torch
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, coo_matrix, save_npz
from sklearn.metrics import f1_score
import gc

from aligned.reasoner import RegulatoryKB

gene_id = pd.read_csv('dataset/human/ensembl_mapping.csv')

device = 'cuda:0'
regu_p_pth = 'scripts/klg_refine/omnipath_P.npz'
regu_n_pth = 'scripts/klg_refine/omnipath_N.npz'

reasoner_true = RegulatoryKB(
        pos_trn_pth=regu_p_pth,
        neg_trn_pth=regu_n_pth,
        device=device)
reasoner_true.closure_(T=5, closure_type='naive')

#print('original KB statics')
#reasoner_true.eval()

n_cols = reasoner_true.KB.shape[1]
X = torch.eye(n_cols).to(device)
Y = reasoner_true.deduce(X)

mean1 = torch.tensor(np.random.uniform(1., 2., size=(len(Y)+1,1))).to(device)
Y1 = torch.concat([.5*Y, torch.zeros(size=(1,Y.shape[1])).to(device)])
Y1 = torch.clip(mean1 + Y1 + torch.tensor(np.random.normal(loc=0, scale=.5, size=Y1.shape)).to(device), .01,)

mean2 = torch.tensor(np.random.uniform(2., 4., size=(len(Y)+1,1))).to(device)
Y2 = torch.concat([1.5*Y, torch.zeros(size=(1,Y.shape[1])).to(device)])
Y2 = torch.clip(mean2 + Y2 + torch.tensor(np.random.normal(loc=0, scale=2., size=Y1.shape)).to(device), .01,)
expr = torch.concat([Y1,Y2]).cpu().numpy()

interv = np.concatenate([np.array(gene_id['gene_id']), np.array(['non-targeting']),\
        np.array(gene_id['gene_id']), np.array(['non-targeting'])])

gene_names = np.array(gene_id['gene_id'])

with open('dataset/human/refine_bench_data.npz', "wb") as file:
    np.savez(
        file,
        expression_matrix= expr,
        var_names= gene_names,
        interventions= interv,
    )
