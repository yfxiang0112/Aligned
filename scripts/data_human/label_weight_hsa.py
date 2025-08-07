import numpy as np
import torch
import pandas as pd
import scanpy as sc
import pickle
from sklearn.metrics import f1_score
from scipy.sparse import load_npz

from egoal.reasoner import RegualtoryKB

data_name = 'norman'

KB = RegualtoryKB(pos_trn_pth=f'dataset/human/{data_name}_KB.npz', neg_trn_pth=None, device='cuda:4')
KB.closure_(T=5, closure_type='diff')

Y = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()
X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()
Y_deduction = KB.deduce(torch.tensor(X).float().to('cuda:4')).to('cpu').numpy()

test_idx = np.load(f'dataset/human/{data_name}_test_idx.npy')
X_test = X[test_idx]
Y_test = Y[test_idx]


#test_df = pd.read_csv(f'dataset/human/{data_name}_test_set.csv', index_col=0)
#series = test_df[test_df['test_type']=='seen_2_pert'].apply(lambda x: list(range(x['data_start_idx'],x['data_end_idx+1'])), axis=1)
#pert_idx = sum(series, [])
#X_test = X_test[pert_idx]
#Y_test = Y_test[pert_idx]

Y_p = np.load('data_anal/abduction_results/Yp_ABL0_hsa.npy')
Y_d = np.load('data_anal/abduction_results/Yd_ABL0_hsa.npy')
R = np.load('data_anal/abduction_results/R_ABL0_hsa.npy')

print(Y_p.shape)

#Y_p = Y_p[pert_idx]
#Y_deduction = Y_deduction[pert_idx]
#R = R[pert_idx]

total = len(Y_test)

#gt_con_idx = (np.nonzero(np.sum((Y_d!= Y_true) | (Y_p != Y_true), axis=0) / total < .2)[0].tolist())
kb_con_idx = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y), axis=0) / total > .2)[0].tolist())

data_idx = np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == np.abs(Y)), axis=1) > 200)[0].tolist()
print(len(data_idx))

#metadata = pd.read_csv(f'dataset/human/{data_name}_metadata.csv',index_col=0)
#metadata_test = metadata.loc[metadata.apply(lambda x: np.any((np.array(data_idx) >= x['data_start_idx']) & (np.array(data_idx) < x['data_end_idx+1'])), axis=1)]
#print(metadata_test)

#Y_test, Y_p, Y_d, R = Y[data_idx], Y_p[data_idx], Y_d[data_idx], R[data_idx]
#Y_p, Y_d, R =  Y_p[test_idx], Y_d[test_idx], R[test_idx]

#print(f'Consistent labels with Y_true\nsra: {len(labels_gt_con)}\n')
print(f'Consistent labels with KB: {len(kb_con_idx)}\n')


' mask index for labels consistent with kb '
mask_kb = np.zeros_like(Y_test, dtype=bool)
mask_kb[:,kb_con_idx] = True
y_mask_kb = np.where(mask_kb, Y_d, Y_p)

y_r = np.where(R, Y_d, Y_p)

print('f1 of Y_p:', f1_score(np.abs(Y_test).flatten(), np.abs(Y_p).flatten(), average='macro'))
print('f1 of Y_deduction:', f1_score(np.abs(Y_test).flatten(), np.abs(Y_d).flatten(), average='macro'))
print('f1 of Y_mask_kb', f1_score(np.abs(Y_test).flatten(), np.abs(y_mask_kb).flatten(), average='macro'))
print('f1 of Y[r]:', f1_score(np.abs(Y_test).flatten(), np.abs(y_r).flatten(), average='macro'))


' weight with GO annotation '
gene2go = pickle.load(open('../GEARS/gene2go_all.pkl', 'rb'))
df_go= pd.read_csv(f'../GEARS/{data_name}/go.csv')
ann_data = sc.read_h5ad(f'../GEARS/{data_name}/perturb_processed.h5ad')
df_genes= pd.DataFrame(ann_data.var)

go_annot_num = np.array([len(gene2go[g]) if g in gene2go else 0 for g in df_genes['gene_name']], dtype=np.float32)
go_annot_num = go_annot_num / np.max(go_annot_num)
#go_annot_num = np.max(go_annot_num - .3, np.zeros_like(go_annot_num))
print(go_annot_num)
y_mask_go = np.where((go_annot_num > .01) & (go_annot_num <= 1.), Y_d, Y_p)
print('f1 of Y_go_weight:', f1_score(Y_test.flatten(), y_mask_go.flatten(), average='macro'))


' weight with in-degree in GRN '
regulatory = load_npz(f'dataset/human/{data_name}_KB.npz').toarray()
regulatory_num = np.sum(regulatory, axis=0)
regulatory_num = regulatory_num / np.max(regulatory_num)
#regulatory_num = np.max(regulatory_num - .3, np.zeros_like(regulatory_num))
print(regulatory_num)
y_mask_regu = np.where((regulatory_num > .01) & (regulatory_num <= 1.), Y_d, Y_p)
print('f1 of Y_grn_weight:', f1_score(Y_test.flatten(), y_mask_regu.flatten(), average='macro'))


' get label weight '
weights = np.full(shape=Y_test.shape[1], fill_value=-.5, dtype=np.float32)
weights += (go_annot_num - .2) + (regulatory_num - .2)
weights[kb_con_idx] += 1.5
weights = np.clip(weights, -1., 1.)
print(weights)
print(np.count_nonzero(weights >= .1))
#print(np.nonzero(weights >= .1)[0].tolist())
np.save(f'dataset/human/{data_name}_label_weight.npy', weights)
