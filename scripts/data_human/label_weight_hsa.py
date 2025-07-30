import numpy as np
import torch
import pandas as pd
import scanpy as sc
import pickle
from sklearn.metrics import f1_score
from scipy.sparse import load_npz

from egoal.reasoner import RegualtoryKB

data_name = 'norman'

KB = RegualtoryKB(pos_trn_pth=f'dataset/human/{data_name}_KB.npz', neg_trn_pth=None, device='cuda')


Y_train = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()
X_train = load_npz(f'dataset/human/{data_name}_X.npz').toarray()
Y_deduction = KB.deduce(torch.tensor(X_train).float().to('cuda'))
Y_deduction = Y_deduction.cpu().numpy()

#Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL2.npy')
#R = np.load('data_anal/abduction_results/R_ABL2.npy')
total = len(Y_train)

#gt_con_idx = (np.nonzero(np.sum((Y_deduction != Y_true) | (Y_pseudo != Y_true), axis=0) / total < .2)[0].tolist())
kb_con_idx = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_train), axis=0) / total > .01)[0].tolist())

#print(f'Consistent labels with Y_true\nsra: {len(labels_gt_con)}\n')
print(f'Consistent labels with KB: {len(kb_con_idx)}\n')


' mask index for labels consistent with kb '
mask_kb = np.zeros_like(Y_train, dtype=bool)
mask_kb[:,kb_con_idx] = True
#y_mask_kb = np.where(mask_kb, Y_deduction, Y_pseudo)

#y_r = np.where(R, Y_deduction, Y_pseudo)

#print('f1 of Y_pseudo:', f1_score(Y_true.flatten(), Y_pseudo.flatten(), average='macro'))
#print('f1 of Y_deduction:', f1_score(Y_true.flatten(), Y_deduction.flatten(), average='macro'))
#print('f1 of Y_mask_kb', f1_score(Y_true.flatten(), y_mask_kb.flatten(), average='macro'))
#print('f1 of Y[r]:', f1_score(Y_true.flatten(), y_r.flatten(), average='macro'))


' weight with GO annotation '
gene2go = pickle.load(open('../GEARS/gene2go_all.pkl', 'rb'))
df_go= pd.read_csv(f'../GEARS/{data_name}/go.csv')
ann_data = sc.read_h5ad(f'../GEARS/{data_name}/perturb_processed.h5ad')
df_genes= pd.DataFrame(ann_data.var)

go_annot_num = np.array([len(gene2go[g]) if g in gene2go else 0 for g in df_genes['gene_name']], dtype=np.float32)
go_annot_num = go_annot_num / np.max(go_annot_num)
#go_annot_num = np.max(go_annot_num - .3, np.zeros_like(go_annot_num))
print(go_annot_num)
#y_mask_go = np.where((go_annot_num > .3) & (go_annot_num <= 1.), Y_deduction, Y_pseudo)
#print('f1 of Y_go_weight:', f1_score(Y_true.flatten(), y_mask_go.flatten(), average='macro'))


' weight with in-degree in GRN '
regulatory = load_npz(f'dataset/human/{data_name}_KB.npz').toarray()
regulatory_num = np.sum(regulatory, axis=0)
regulatory_num = regulatory_num / np.max(regulatory_num)
#regulatory_num = np.max(regulatory_num - .3, np.zeros_like(regulatory_num))
print(regulatory_num)
#y_mask_regu = np.where((regulatory_num > .3) & (regulatory_num <= 1.), Y_deduction, Y_pseudo)
#print('f1 of Y_grn_weight:', f1_score(Y_true.flatten(), y_mask_regu.flatten(), average='macro'))


' get label weight '
weights = np.full(shape=Y_train.shape[1], fill_value=-.3, dtype=np.float32)
weights += (go_annot_num - .3) + (regulatory_num - .3)
weights[kb_con_idx] += 1.3
weights = np.clip(weights, -1., 1.)
print(weights)
print(np.count_nonzero(weights >= .1))
np.save(f'dataset/human/{data_name}_label_weight.npy', weights)
