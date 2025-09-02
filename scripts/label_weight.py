import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.sparse import load_npz
import torch

from egoal.reasoner import RegulatoryKB
from egoal.learner_refl import ReflectLearner

label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
idx_list_p1k = list(label_set['precise1k_idx'])
idx_list_sra = list(label_set['matrix_idx'])


X_p1k = np.load('dataset/precise1k/X_label.npy')
Y_p1k = np.load('dataset/precise1k/Y_label.npy')

X_sra = np.load('dataset/ncbi-sra/X_label.npy')
Y_sra = np.load('dataset/ncbi-sra/Y_label.npy')

Y_p1k = Y_p1k[:,idx_list_p1k]
Y_sra = Y_sra[:,idx_list_sra]

test_idx_p1k = np.zeros(len(X_p1k), dtype=bool)
test_idx_p1k[[ 280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323]] = True

test_idx_sra = np.zeros(len(X_sra), dtype=bool)
test_idx_sra[[37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]] = True
# arcZ, gcvB, micA, ryhB

X_test = np.concat([X_p1k[test_idx_p1k], X_sra[test_idx_sra]])
Y_test = np.concat([Y_p1k[test_idx_p1k], Y_sra[test_idx_sra]])

#label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)

#' precise1k '

#Y_true = np.load('dataset/precise1k/Y_label.npy')[:,list(label_set['precise1k_idx'])]
#Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0_p1k.npy')
#Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0_p1k.npy')
#total = len(Y_true)


#' less inconsistencies & mistakes on data '
#labels_gt_con_p1k = (np.nonzero(np.sum((Y_deduction != Y_true) | (Y_pseudo != Y_true), axis=0) / total < .2)[0].tolist())

#' klg consistency '
#labels_kb_con_p1k = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_true), axis=0) / total > .1)[0].tolist())

#labels_zero_p1k = np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9)
#labels_nonzero_p1k = np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .05)

#########################

#' ncbi-sra '

#test_idx = [37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]
#Y_true = np.load('dataset/ncbi-sra/Y_label.npy')[test_idx][:,list(label_set['matrix_idx'])]
#Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0.npy')
#Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0.npy')
#R = np.load('data_anal/abduction_results/R_ABL0.npy')
#total = len(Y_true)

#labels_gt_con_sra = (np.nonzero(np.sum((Y_deduction != Y_true) | (Y_pseudo != Y_true), axis=0) / total < .2)[0].tolist())
#labels_kb_con_sra = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_true), axis=0) / total > .1)[0].tolist())


#label_zeros_sra = np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9)
#label_nonzero_sra = np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .3)

#print(f'Consistent labels with Y_true\np1k: {len(labels_gt_con_p1k)}, sra: {len(labels_gt_con_sra)}, intersec: {len(set(labels_gt_con_p1k).intersection(set(labels_gt_con_sra)))}\n')

#print(f'Consistent labels with KB\np1k: {len(labels_kb_con_p1k)}, sra: {len(labels_kb_con_sra)}, intersec: {len(set(labels_kb_con_p1k).intersection(set(labels_kb_con_sra)))}\n')



device = 'cuda'

KB = RegulatoryKB(pos_trn_pth=f'rules/regu_pos.npz', neg_trn_pth=f'rules/regu_neg.npz', output_idx_list=idx_list_sra, device=device)
KB.closure_(T=5, closure_type='weighted')


Y_d = KB.deduce(torch.tensor(X_test).float().to(device)).to('cpu').numpy()

#adj_matrix = torch.round(torch.clamp(torch.abs(KB.Regu_N_0 + KB.Regu_P_0), 0,1))
adj_matrix = torch.round(torch.abs(KB.KB))
learner = ReflectLearner(input_dim= X_test.shape[1],
                         output_dim= Y_test.shape[1],
                         hidden_dim= 64,
                         r_layer= True,
                         base_learner_type= 'GNN',
                         adj_matrix= adj_matrix,
                         device=device)
learner.load('models/GNN_eco_Sep1.pt')

Y_p = learner.predict(torch.tensor(X_test).float().to(device)).to('cpu').numpy()
R = learner.reflection(torch.tensor(X_test).float().to(device)).to('cpu').numpy().astype(bool)


' mask index for labels consistent with kb '
total = len(Y_test)
mask_kb = np.zeros_like(Y_test, dtype=bool)
#kb_con_idx = list(set(labels_kb_con_p1k).intersection(set(labels_kb_con_sra)))
kb_con_idx = (np.nonzero(np.sum((Y_d!= 0) & (Y_d == Y_test), axis=0) / total > .1)[0].tolist())
print(f'consistent labels: {len(kb_con_idx)}')
mask_kb[:,kb_con_idx] = True
y_mask_kb = np.where(mask_kb, Y_d, Y_p)

#R = np.load('R_ABL0.npy')
y_r = np.where(R, Y_d, Y_p)

print('f1 of Y_pseudo:', f1_score(Y_test.flatten(), Y_p.flatten(), average='macro'))
print('f1 of Y_deduction:', f1_score(Y_test.flatten(), Y_d.flatten(), average='macro'))
print('f1 of Y_mask_kb', f1_score(Y_test.flatten(), y_mask_kb.flatten(), average='macro'))
print('f1 of Y[r]:', f1_score(Y_test.flatten(), y_r.flatten(), average='macro'))
#
#labels_r = np.nonzero(np.sum(R, axis=0) > 5)[0].tolist()
#print(len(labels_r))
#print(len(idx))
#
#print(len(set(labels_r) - set(idx)))
#print(len(set(idx) - set(labels_r)))
#
#print(np.count_nonzero(np.sum(Y_deduction, axis=1) / total > .5))
#


' weight with GO annotation '
gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
go_annot = pd.read_csv('rules/GO/goa_gene2go.csv', index_col = 0)
#go_annot = go_annot.loc[gene_idx.loc[gene_idx['iml1515_idx']!=-1, 'locus']]
go_annot_num = np.array([len(eval(go_annot.loc[i,'concepts'])) if i in go_annot.index else 0 for i in gene_idx.loc[gene_idx['iml1515_idx']!=-1, 'locus']], dtype=np.float32)
go_annot_num /= np.max(go_annot_num)
#go_annot_num = np.max(go_annot_num - .3, np.zeros_like(go_annot_num))
print(go_annot_num)
y_mask_go = np.where((go_annot_num > .3) & (go_annot_num <= 1.), Y_d, Y_p)
print('f1 of Y_go_weight:', f1_score(Y_test.flatten(), y_mask_go.flatten(), average='macro'))


' weight with in-degree in GRN '
iml_idx = list(gene_idx[gene_idx['iml1515_idx']!=-1].index)
regulatory_p = load_npz('rules/regu_pos.npz').toarray()
regulatory_n = load_npz('rules/regu_neg.npz').toarray()
regulatory_num = np.sum(regulatory_p + regulatory_n, axis=0)[iml_idx]
regulatory_num /= np.max(regulatory_num)
#regulatory_num = np.max(regulatory_num - .3, np.zeros_like(regulatory_num))
print(regulatory_num)
y_mask_regu = np.where((regulatory_num > .3) & (regulatory_num <= 1.), Y_d, Y_p)
print('f1 of Y_grn_weight:', f1_score(Y_test.flatten(), y_mask_regu.flatten(), average='macro'))


' get label weight '
weights = np.full(shape=Y_test.shape[1], fill_value=.1, dtype=np.float32)
weights += (go_annot_num - .3) + (regulatory_num - .35)
weights[kb_con_idx] += .8
weights = np.clip(weights, 0., 1.)
print(weights)
print(np.count_nonzero(weights >= .5))
np.save('rules/label_weight.npy', weights)

Y_w = np.where(weights>=.5, Y_d, Y_p)
f1_test = f1_score(Y_test.flatten(), Y_w.flatten(), average='macro')
print(f'label weight f1: {f1_test}')
