import torch
import numpy as np
import pandas as pd
from aligned.reasoner import RegulatoryKB
from aligned.learner_adap import AdaptorLearner
from scipy.sparse import coo_matrix, load_npz, save_npz
import json

seed = 88
data_name = 'norman'
device = 'cuda:0'
np.random.seed(seed)

gene_ann = pd.read_csv(f'dataset/human/{data_name}_gene_ann.csv', index_col='vector_idx')
df_omnipath = pd.read_csv('rules/human/omnipath.csv', index_col=0)

xref_string = load_npz('data_anal/xref_csbench/networks/string.npz').toarray()
xref_corum = load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()
xref_lr = load_npz('data_anal/xref_csbench/networks/lr_pairs.npz').toarray()
xref_chipseq = load_npz('data_anal/xref_csbench/networks/chipseq.npz').toarray()

omnipath_conf = load_npz('data_anal/xref_csbench/networks/omni_conf.npz').toarray()

omnipath_P = load_npz(f'data_anal/xref_csbench/networks/{data_name}_omnipath_KB_P.npz').toarray()
omnipath_N = load_npz(f'data_anal/xref_csbench/networks/{data_name}_omnipath_KB_N.npz').toarray()

corr_P = load_npz(f'data_anal/xref_csbench/networks/{data_name}_Corr_P.npz').toarray()
corr_N = load_npz(f'data_anal/xref_csbench/networks/{data_name}_Corr_N.npz').toarray()

confirm_P = ((xref_string!=0)|(xref_corum!=0)) & (xref_chipseq!=0) & (omnipath_P!=0) & (omnipath_conf > 5)
confirm_N = ((xref_string!=0)|(xref_corum!=0)) & (xref_chipseq!=0) & (omnipath_N!=0) & (omnipath_conf > 5)

sel_in_data = (confirm_P & (corr_P > 2.)) | (confirm_N & (corr_N > 2.))
print(f'{np.count_nonzero(sel_in_data)} confirmed interactions in data corr')
p_rand = min(1., 10/np.count_nonzero(sel_in_data))
sel_in_data = sel_in_data & np.random.choice([True, False], size=(sel_in_data.shape[0],sel_in_data.shape[1]), p=[p_rand, 1-p_rand])
print(f'{np.count_nonzero(sel_in_data)} selected interactions in data corr')

sel_ex_data = (confirm_P & (corr_P < .1) & (corr_P > 0)) | (confirm_N & (corr_N < .1) & (corr_N > 0))
print(f'{np.count_nonzero(sel_ex_data)} confirmed interactions not in data corr')
p_rand = min(1., 10/np.count_nonzero(sel_ex_data))
sel_ex_data = sel_ex_data & np.random.choice([True, False], size=(sel_in_data.shape[0],sel_in_data.shape[1]), p=[p_rand, 1-p_rand])
print(f'{np.count_nonzero(sel_ex_data)} selected interactions not in data corr')

selected = sel_in_data | sel_ex_data

res_regulations = []
for i,j in zip(*np.nonzero(selected)):
    src = gene_ann.loc[i,'gene_name']
    tar = gene_ann.loc[j,'gene_name']
    #print(src,tar)

    res_regulations.append(df_omnipath[(df_omnipath['source']==src) & (df_omnipath['target']==tar)])
    res_regulations[-1].insert(loc=2, column='data_corr', value=[max(corr_P[i,j],corr_N[i,j])]*len(res_regulations[-1]))

df_selected = pd.concat(res_regulations,axis=0)
print(df_selected)
df_selected.to_csv(f'data_anal/xref_csbench/recovery_KB/{data_name}_interactions_{seed}.csv')

KB_P = load_npz(f'rules/human/{data_name}_KB_P.npz').toarray()
KB_N = load_npz(f'rules/human/{data_name}_KB_N.npz').toarray()
KB_P = np.where(selected, 0, KB_P)
KB_N = np.where(selected, 0, KB_N)
save_npz(f'data_anal/xref_csbench/recovery_KB/{data_name}_KB_P_{seed}.npz', coo_matrix(KB_P))
save_npz(f'data_anal/xref_csbench/recovery_KB/{data_name}_KB_N_{seed}.npz', coo_matrix(KB_N))
