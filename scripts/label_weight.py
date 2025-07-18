import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.sparse import load_npz

test_idx = [37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]
label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)

' precise1k '

Y_test = np.load('dataset/precise1k/Y_label.npy')[:,list(label_set['precise1k_idx'])]

Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0_p1k.npy')
Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0_p1k.npy')

total = len(Y_test)

#print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_test == 0), axis=0) / total > .7))
#print(np.count_nonzero(np.sum((Y_pseudo == 0) & (Y_test == 0), axis=0) / total > .7))
#
#print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_test == Y_deduction), axis=0) / total > .025))
#print(np.count_nonzero(np.sum((Y_pseudo != 0) & (Y_test == Y_pseudo), axis=0) / total > .1))


labels_p1k = (np.nonzero(np.sum((Y_deduction != Y_test) | (Y_pseudo != Y_test), axis=0) / total < .2)[0].tolist())
labels_p1k_deduc= (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_test), axis=0) / total > .01)[0].tolist())


print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9))
print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .05))

#########################

' ncbi-sra '

Y_test = np.load('dataset/ncbi-sra/Y_label.npy')[test_idx][:,list(label_set['matrix_idx'])]

Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0_regulator.npy')
Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0_regulator.npy')

total = len(Y_test)

#print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_test == 0), axis=0) / total > .7))
#print(np.count_nonzero(np.sum((Y_pseudo == 0) & (Y_test == 0), axis=0) / total > .7))
#
#print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_test == Y_deduction), axis=0) / total > .2))
#print(np.count_nonzero(np.sum((Y_pseudo != 0) & (Y_test == Y_pseudo), axis=0) / total > .3))

labels_sra = (np.nonzero(np.sum((Y_deduction != Y_test) | (Y_pseudo != Y_test), axis=0) / total < .2)[0].tolist())
labels_sra_deduc= (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_test), axis=0) / total > .1)[0].tolist())


print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9))
print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .3))

print(len(labels_p1k), len(labels_sra))
print(len(set(labels_p1k).intersection(set(labels_sra))))

print(len(labels_p1k_deduc), len(labels_sra_deduc))
print(len(set(labels_p1k_deduc).intersection(set(labels_sra_deduc))))

#print(sorted(list(set(labels_p1k).intersection(set(labels_sra)).union(set(labels_p1k_deduc).intersection(set(labels_sra_deduc))))))


##NOTE tmp
mask = np.zeros_like(Y_test, dtype=bool)
deduction_idx = list(set(labels_p1k_deduc).intersection(set(labels_sra_deduc)))
#print(sorted(idx))
mask[:,deduction_idx] = True
y_mask = np.where(mask, Y_deduction, Y_pseudo)

#R = np.load('R_ABL0.npy')
R = np.load('data_anal/abduction_results/R_ABL0_regulator.npy')
y_r = np.where(R, Y_deduction, Y_pseudo)

print(f1_score(Y_test.flatten(), Y_pseudo.flatten(), average='macro'))
print(f1_score(Y_test.flatten(), Y_deduction.flatten(), average='macro'))
print(f1_score(Y_test.flatten(), y_mask.flatten(), average='macro'))
print(f1_score(Y_test.flatten(), y_r.flatten(), average='macro'))
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


gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
go_annot = pd.read_csv('rules/GO/goa_gene2go.csv', index_col = 0)
#go_annot = go_annot.loc[gene_idx.loc[gene_idx['iml1515_idx']!=-1, 'locus']]
go_annot_num = np.array([len(eval(go_annot.loc[i,'concepts'])) if i in go_annot.index else 0 for i in gene_idx.loc[gene_idx['iml1515_idx']!=-1, 'locus']], dtype=np.float32)
go_annot_num /= np.max(go_annot_num)
#go_annot_num = np.max(go_annot_num - .3, np.zeros_like(go_annot_num))
print(go_annot_num)
y_mask_go = np.where((go_annot_num > .3) & (go_annot_num <= 1.), Y_deduction, Y_pseudo)
print(f1_score(Y_test.flatten(), y_mask_go.flatten(), average='macro'))


iml_idx = list(gene_idx[gene_idx['iml1515_idx']!=-1].index)
regulatory_p = load_npz('rules/regu_pos.npz').toarray()
regulatory_n = load_npz('rules/regu_neg.npz').toarray()
regulatory_num = np.sum(regulatory_p + regulatory_n, axis=0)[iml_idx]
regulatory_num /= np.max(regulatory_num)
#regulatory_num = np.max(regulatory_num - .3, np.zeros_like(regulatory_num))
print(regulatory_num)
y_mask_regu = np.where((regulatory_num > .3) & (regulatory_num <= 1.), Y_deduction, Y_pseudo)
print(f1_score(Y_test.flatten(), y_mask_regu.flatten(), average='macro'))


weights = np.full(shape=Y_test.shape[1], fill_value=-.3, dtype=np.float32)
weights += (go_annot_num - .3) + (regulatory_num - .3)
weights[deduction_idx] += 1.3
weights = np.clip(weights, -1., 1.)
print(weights)
print(np.count_nonzero(weights >= .1))
np.save('rules/label_weight.npy', weights)
