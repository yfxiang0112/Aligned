import torch
import numpy as np
from egoal.reasoner import RegulatoryKB
from egoal.learner_refl import ReflectLearner
from scipy.sparse import load_npz
import json
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

data_name = 'norman'
device = 'cuda:0'

xref_string = load_npz('data_anal/xref_csbench/networks/string.npz').toarray()
xref_string = torch.tensor(xref_string).to(device)
xref_corum = load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()
xref_corum = torch.tensor(xref_corum).to(device)
xref_lr = load_npz('data_anal/xref_csbench/networks/lr_pairs.npz').toarray()
xref_lr = torch.tensor(xref_lr).to(device)
xref_chipseq = load_npz('data_anal/xref_csbench/networks/chipseq.npz').toarray()
xref_chipseq = torch.tensor(xref_chipseq).to(device)

omni_conf = load_npz('data_anal/xref_csbench/networks/omni_conf.npz').toarray()
omni_conf = torch.tensor(omni_conf).to(device)
#omni_conf = (omni_conf>10).int() #TODO

n_genes = xref_string.shape[1]
X = torch.eye(n_genes).to(device)

reasoner = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device=device)
reasoner.closure_(T=5, closure_type='weighted')

adj_matrix = torch.round(torch.clamp(torch.abs(reasoner.Regu_N_0 + reasoner.Regu_P_0), 0,1))
learner = ReflectLearner(input_dim= n_genes,
                         output_dim= n_genes,
                         hidden_dim= 64,
                         base_learner_type= 'GNN',
                         adj_matrix= adj_matrix,
                         device=device)

label_weight = torch.tensor(np.load(f'dataset/human/{data_name}_label_weight.npy')).to(device)
print('w>.5: ', torch.count_nonzero(label_weight>=.5).item())

for repl in range(5):
    reasoner.load(f'data_anal/experiment_results/{data_name}/models/GNN_abl0_{repl+1}.npz')
    learner.load(f'data_anal/experiment_results/{data_name}/models/GNN_abl0_{repl+1}.pt')
    
    Y_p = learner.predict(X).float()
    R = learner.reflection(X)
    Y_d= reasoner.deduce(torch.tensor(X).float().to(device)).float()

    for ref, name in [(xref_string, 'string'),
                      (xref_corum,  'corum'),
                      (xref_lr,     'lr'),
                      (xref_chipseq,'chipseq')]:

        print(f'\n{name}:')
        R0 = torch.count_nonzero((ref==0)&(Y_d==0) & (R==0)).item()
        R1 = torch.count_nonzero((ref==0)&(Y_d==0) & (R==1)).item()
        print(f'TN:    R=0 {R0/(R0+R1):.3f}, R=1 {R1/(R0+R1):.3f}')

        R0 = torch.count_nonzero((ref!=0)&(Y_d!=0) & (R==0)).item()
        R1 = torch.count_nonzero((ref!=0)&(Y_d!=0) & (R==1)).item()
        print(f'TP:    R=0 {R0/(R0+R1):.3f}, R=1 {R1/(R0+R1):.3f}')

        R0 = torch.count_nonzero(((ref!=0)&(Y_d==0) | (ref==0)&(Y_d!=0)) & (R==0)).item()
        R1 = torch.count_nonzero(((ref!=0)&(Y_d==0) | (ref==0)&(Y_d!=0)) & (R==1)).item()
        print(f'FN&FP: R=0 {R0/(R0+R1):.3f}, R=1 {R1/(R0+R1):.3f}')

        #R0 = torch.count_nonzero((ref==0)&(Y_d!=0) & (R==0)).item()
        #R1 = torch.count_nonzero((ref==0)&(Y_d!=0) & (R==1)).item()
        #print(f'FP: R=0 {R0/(R0+R1): .3f}, R=1 {R1/(R0+R1): .3f}')

    print('\nomnipath conf:')
    R0 = torch.count_nonzero((omni_conf==0) & (R==0)).item()
    R1 = torch.count_nonzero((omni_conf==0) & (R==1)).item()
    print(f'c==0:     R=0 {R0/(R0+R1): .3f}, R=1 {R1/(R0+R1): .3f}')

    R0 = torch.count_nonzero((omni_conf>=1)&(omni_conf<5) & (R==0)).item()
    R1 = torch.count_nonzero((omni_conf>=1)&(omni_conf<5) & (R==1)).item()
    print(f'1<=c<5:   R=0 {R0/(R0+R1): .3f}, R=1 {R1/(R0+R1): .3f}')

    R0 = torch.count_nonzero((omni_conf>=5)&(omni_conf<10) & (R==0)).item()
    R1 = torch.count_nonzero((omni_conf>=5)&(omni_conf<10) & (R==1)).item()
    print(f'5<=c<10:  R=0 {R0/(R0+R1): .3f}, R=1 {R1/(R0+R1): .3f}')

    R0 = torch.count_nonzero((omni_conf>=10)&(omni_conf<20) & (R==0)).item()
    R1 = torch.count_nonzero((omni_conf>=10)&(omni_conf<20) & (R==1)).item()
    print(f'10<=c<20: R=0 {R0/(R0+R1): .3f}, R=1 {R1/(R0+R1): .3f}')

    R0 = torch.count_nonzero((omni_conf>=20) & (R==0)).item()
    R1 = torch.count_nonzero((omni_conf>=20) & (R==1)).item()
    print(f'c>=20:    R=0 {R0/(R0+R1): .3f}, R=1 {R1/(R0+R1): .3f}')


    #print(torch.count_nonzero((xref_string==0)&(Y_d!=0)&(Y_d!=Y_p) & (R==0)))
    #print(torch.count_nonzero((xref_string==0)&(Y_d!=0)&(Y_d!=Y_p) & (R==1)))
    exit()
