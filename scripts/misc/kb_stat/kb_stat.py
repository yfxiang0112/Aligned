import torch
import numpy as np
from aligned.reasoner import RegulatoryKB
from aligned.learner_adap import AdaptorLearner
from scipy.sparse import load_npz
import json

data_name = 'norman'
device = 'cuda'

Y = torch.tensor(load_npz(f'dataset/human/{data_name}_Y.npz').toarray()).to(device).float()
X = torch.tensor(load_npz(f'dataset/human/{data_name}_X.npz').toarray()).to(device).float()

results = { 'n_consit_y1_kb0':[],
        'n_consit_y0_kb1':[],
        'n_consit_y1_kb1':[],
        'n_incomp_y1_kb0':[],
        'n_incomp_y0_kb1':[],
        'n_incomp_y1_kb1':[],
        'n_incons_y1_kb0':[],
        'n_incons_y0_kb1':[],
        'n_incons_y1_kb1':[],
        'modularity_0':   [],
        'assortativity_0':[],
        'modularity_1':   [],
        'assortativity_1':[],
        'modularity_2':   [],
        'assortativity_2':[],
        'gsr_+':          [],
        'gsr_-':          [],
        'gsr_+_2':        [],
        'gsr_-_2':        [],
        }

reasoner = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device=device)
scores_before = reasoner.eval()
reasoner.closure_(T=5, closure_type='weighted')
KB_before = reasoner.KB
results['assortativity_0'].append(scores_before['degree_assortativity'])
results['modularity_0'].append(scores_before['modularity'])


adj_matrix = torch.round(torch.clamp(torch.abs(reasoner.Regu_N_0 + reasoner.Regu_P_0), 0,1))
learner = AdaptorLearner(input_dim= X.shape[1],
                         output_dim= Y.shape[1],
                         hidden_dim= 64,
                         base_learner_type= 'GNN',
                         adj_matrix= adj_matrix,
                         device=device)

unique_X, inverse_indices = torch.unique(X, dim=0, return_inverse=True)
Y_means_P = torch.zeros((len(unique_X), Y.shape[1])).to(device)
Y_means_N = torch.zeros((len(unique_X), Y.shape[1])).to(device)
for i in range(len(unique_X)):
    mask = (inverse_indices == i)
    Y_means_P[i] = torch.mean(torch.clamp(Y,min=0)[mask], dim=0)
    Y_means_N[i] = torch.mean(torch.clamp(-Y,min=0)[mask], dim=0)

corr_before_P = (torch.clamp(unique_X,min=0).T @ Y_means_P)\
        + (torch.clamp(-unique_X,min=0).T @ Y_means_N)
corr_before_N = (torch.clamp(unique_X,min=0).T @ Y_means_N)\
        + (torch.clamp(-unique_X,min=0).T @ Y_means_P)
del Y

gsr_orig = json.load(open('scripts/misc/net_eval/results/kegg_orig_norman.json', 'r'))

for repl in range(5):
    learner.load(f'results/ex1_aligned/{data_name}/models/GNN_abl0_{repl+1}.pt')
    
    reasoner.load(f'results/ex1_aligned/{data_name}/models/GNN_abl0_{repl+1}.npz')
    scores_after = reasoner.eval()
    KB_after = reasoner.KB

    results['assortativity_1'].append(scores_after['degree_assortativity'])
    results['modularity_1'].append(scores_after['modularity'])

    
    Y_p = learner.predict(torch.tensor(X).float().to(device)).float()
    R = learner.reflection(torch.tensor(X).float().to(device)) >= .5
    Y_d= reasoner.deduce(torch.tensor(X).float().to(device)).float()
    Y_r = torch.where(R, Y_d, Y_p).float()
    
    Yr_means_P = torch.zeros((len(unique_X), Y_r.shape[1])).to(device)
    Yr_means_N = torch.zeros((len(unique_X), Y_r.shape[1])).to(device)
    for i in range(len(unique_X)):
        mask = (inverse_indices == i)
        Yr_means_P[i] = torch.mean(torch.clamp(Y_r,min=0)[mask], dim=0)
        Yr_means_N[i] = torch.mean(torch.clamp(-Y_r,min=0)[mask], dim=0)
    
    corr_r_P = (torch.clamp(unique_X,min=0).T @ Yr_means_P)\
            + (torch.clamp(-unique_X,min=0).T @ Yr_means_N)
    corr_r_N = (torch.clamp(unique_X,min=0).T @ Yr_means_N)\
            + (torch.clamp(-unique_X,min=0).T @ Yr_means_P)

    n_consit_y1_kb0 = int(torch.sum(corr_r_P[KB_before>0]) + torch.sum(corr_r_N[KB_before<0]))
    n_incomp_y1_kb0 = int(torch.sum((corr_r_P+corr_r_N)[KB_before==0]))
    n_incons_y1_kb0 = int(torch.sum((1-corr_r_P)[KB_before>0]) + torch.sum((1-corr_r_N)[KB_before<0]))
    n_empty_y1_kb0 = int(torch.sum((1-corr_r_P-corr_r_N)[KB_before==0]))
    total = n_consit_y1_kb0 + n_incomp_y1_kb0 + n_incons_y1_kb0 + n_empty_y1_kb0
    n_consit_y1_kb0, n_incomp_y1_kb0, n_incons_y1_kb0, n_empty_y1_kb0 =\
            n_consit_y1_kb0/total, n_incomp_y1_kb0/total, n_incons_y1_kb0/total, n_empty_y1_kb0/total
    results['n_consit_y1_kb0'].append(n_consit_y1_kb0)
    results['n_incomp_y1_kb0'].append(n_incomp_y1_kb0)
    results['n_incons_y1_kb0'].append(n_incons_y1_kb0)

    n_consit_y0_kb1 = int(torch.sum(corr_r_P[KB_after>0]) + torch.sum(corr_r_N[KB_after<0]))
    n_incomp_y0_kb1 = int(torch.sum((corr_r_P+corr_r_N)[KB_after==0]))
    n_incons_y0_kb1 = int(torch.sum((1-corr_r_P)[KB_after>0]) + torch.sum((1-corr_r_N)[KB_after<0]))
    n_empty_y0_kb1 = int(torch.sum((1-corr_r_P-corr_r_N)[KB_after==0]))
    total = n_consit_y0_kb1 + n_incomp_y0_kb1 + n_incons_y0_kb1 + n_empty_y0_kb1
    n_consit_y0_kb1, n_incomp_y0_kb1, n_incons_y0_kb1, n_empty_y0_kb1 = n_consit_y0_kb1/total, n_incomp_y0_kb1/total, n_incons_y0_kb1/total, n_empty_y0_kb1/total
    results['n_consit_y0_kb1'].append(n_consit_y0_kb1)
    results['n_incomp_y0_kb1'].append(n_incomp_y0_kb1)
    results['n_incons_y0_kb1'].append(n_incons_y0_kb1)

    n_consit_y1_kb1 = int(torch.sum(corr_r_P[KB_after>0]) + torch.sum(corr_r_N[KB_after<0]))
    n_incomp_y1_kb1 = int(torch.sum((corr_r_P+corr_r_N)[KB_after==0]))
    n_incons_y1_kb1 = int(torch.sum((1-corr_r_P)[KB_after>0]) + torch.sum((1-corr_r_N)[KB_after<0]))
    n_empty_y1_kb1 = int(torch.sum((1-corr_r_P-corr_r_N)[KB_after==0]))
    total = n_consit_y1_kb1 + n_incomp_y1_kb1 + n_incons_y1_kb1 + n_empty_y1_kb1
    n_consit_y1_kb1, n_incomp_y1_kb1, n_incons_y1_kb1, n_empty_y1_kb1 =\
            n_consit_y1_kb1/total, n_incomp_y1_kb1/total, n_incons_y1_kb1/total, n_empty_y1_kb1/total
    results['n_consit_y1_kb1'].append(n_consit_y1_kb1)
    results['n_incomp_y1_kb1'].append(n_incomp_y1_kb1)
    results['n_incons_y1_kb1'].append(n_incons_y1_kb1)

    del Y_r
    del R
    del Y_d
    del corr_r_P
    del corr_r_N
    torch.cuda.empty_cache()

    gsr_refine = json.load(open(f'scripts/misc/net_eval/results/kegg_norman_abl0_{repl+1}.json', 'r'))
    results['gsr_+'].append(len([k for k in gsr_refine.keys() if gsr_refine[k]['AUPRC_pos'] > gsr_orig[k]['AUPRC_pos']]))
    results['gsr_-'].append(len([k for k in gsr_refine.keys() if gsr_refine[k]['AUPRC_pos'] < gsr_orig[k]['AUPRC_pos']]))

    reasoner.load(f'results/ex1_aligned/{data_name}/models/GNN_abl1_{repl+1}.npz')
    scores_after = reasoner.eval()
    results['assortativity_2'].append(scores_after['degree_assortativity'])
    results['modularity_2'].append(scores_after['modularity'])

    gsr_refine = json.load(open(f'scripts/misc/net_eval/results/kegg_norman_abl1_{repl+1}.json', 'r'))
    results['gsr_+_2'].append(len([k for k in gsr_refine.keys() if gsr_refine[k]['AUPRC_pos'] > gsr_orig[k]['AUPRC_pos']]))
    results['gsr_-_2'].append(len([k for k in gsr_refine.keys() if gsr_refine[k]['AUPRC_pos'] < gsr_orig[k]['AUPRC_pos']]))

results_std = {}
for k,v in results.items():
    results_std[k] = [.5*(max(v)+min(v)), .5*(max(v)-min(v))]
json.dump(results_std, open('scripts/misc/kb_stat/kb_stat.json', 'w'), indent=4)
