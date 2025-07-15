import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

test_idx = [37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]
label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)

' precise1k '

Y_test = np.load('dataset/precise1k/Y_label.npy')[:,list(label_set['precise1k_idx'])]

Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0_p1k.npy')
Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0_p1k.npy')

total = len(Y_test)

print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_test == 0), axis=0) / total > .7))
print(np.count_nonzero(np.sum((Y_pseudo == 0) & (Y_test == 0), axis=0) / total > .7))

print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_test == Y_deduction), axis=0) / total > .025))
print(np.count_nonzero(np.sum((Y_pseudo != 0) & (Y_test == Y_pseudo), axis=0) / total > .1))


labels_p1k = (np.nonzero(np.sum((Y_deduction != Y_test) | (Y_pseudo != Y_test), axis=0) / total < .2)[0].tolist())
labels_p1k_deduc= (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_test), axis=0) / total > .01)[0].tolist())


print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9))
print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .05))

#########################

' ncbi-sra '

Y_test = np.load('dataset/ncbi-sra/Y_label.npy')[test_idx][:,list(label_set['matrix_idx'])]

Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0.npy')
Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0.npy')

total = len(Y_test)

print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_test == 0), axis=0) / total > .7))
print(np.count_nonzero(np.sum((Y_pseudo == 0) & (Y_test == 0), axis=0) / total > .7))

print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_test == Y_deduction), axis=0) / total > .2))
print(np.count_nonzero(np.sum((Y_pseudo != 0) & (Y_test == Y_pseudo), axis=0) / total > .3))

labels_sra = (np.nonzero(np.sum((Y_deduction != Y_test) | (Y_pseudo != Y_test), axis=0) / total < .2)[0].tolist())
labels_sra_deduc= (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_test), axis=0) / total > .1)[0].tolist())


print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9))
print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .3))

print(len(labels_p1k), len(labels_sra))
print(len(set(labels_p1k).intersection(set(labels_sra))))

print(len(labels_p1k_deduc), len(labels_sra_deduc))
print(len(set(labels_p1k_deduc).intersection(set(labels_sra_deduc))))

#print(sorted(list(set(labels_p1k).intersection(set(labels_sra)).union(set(labels_p1k_deduc).intersection(set(labels_sra_deduc))))))


#NOTE tmp
mask = np.zeros_like(Y_test, dtype=bool)
idx = list(set(labels_p1k_deduc).intersection(set(labels_sra_deduc)))
#print(sorted(idx))
mask[:,idx] = True
y = np.where(mask, Y_deduction, Y_pseudo)
print(f1_score(Y_test.flatten(), Y_pseudo.flatten(), average='macro'))
print(f1_score(Y_test.flatten(), Y_deduction.flatten(), average='macro'))
print(f1_score(Y_test.flatten(), y.flatten(), average='macro'))

R = np.load('data_anal/abduction_results/R_ABL0.npy')
labels_r = np.nonzero(np.sum(R, axis=0) > 5)[0].tolist()
print(len(labels_r))
print(len(idx))

print(len(set(labels_r) - set(idx)))
print(len(set(idx) - set(labels_r)))
