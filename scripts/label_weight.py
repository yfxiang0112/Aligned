import numpy as np
import pandas as pd

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


print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9))
print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .05))

#########################

' ncbi-sra '

Y_test = np.load('dataset/ncbi-sra/Y_label.npy')[test_idx][:,list(label_set['matrix_idx'])]

Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0_regulator.npy')
Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0_regulator.npy')

total = len(Y_test)

print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_test == 0), axis=0) / total > .7))
print(np.count_nonzero(np.sum((Y_pseudo == 0) & (Y_test == 0), axis=0) / total > .7))

print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_test == Y_deduction), axis=0) / total > .3))
print(np.count_nonzero(np.sum((Y_pseudo != 0) & (Y_test == Y_pseudo), axis=0) / total > .3))
print(np.nonzero(np.sum((Y_deduction != 0) & (Y_test == Y_deduction), axis=0) / total > .3)[0].tolist())


print(np.count_nonzero(np.sum((Y_deduction == 0) & (Y_pseudo == 0), axis=0) / total > .9))
print(np.count_nonzero(np.sum((Y_deduction != 0) & (Y_pseudo != 0), axis=0) / total > .3))
