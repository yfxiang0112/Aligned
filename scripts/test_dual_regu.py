import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix, f1_score

def closure(R_P_0, R_N_0, T=None, device='cpu'):
    cnt = 1
    R_P_0, R_N_0 = R_P_0.to(device), R_N_0.to(device)
    I = torch.eye(R_P_0.shape[0]).to(device)
    
    R_P, R_N = R_P_0+I, R_N_0
    while True:
        if T!=None and cnt >= T:
            break

        R_P_, R_N_ = R_P, R_N
    
        R_P = R_P_0 @ R_P_ + R_N_0 @ R_N_
        R_N = R_P_0 @ R_N_ + R_N_0 @ R_P_
    
        #R_P, R_N = torch.clamp(R_P,0,1), torch.clamp(R_N,0,1)
        R_P /= torch.min(R_P[R_P!=0])
        R_N /= torch.min(R_N[R_N!=0])
    
        if torch.all(R_P == R_P_) and torch.all(R_N == R_N_):
            break
        cnt += 1
    
    R_N = torch.where(I.bool(), 0, R_N)
    return R_P, R_N


device = 'cuda'
R_P_0 = torch.tensor(load_npz('rules/regu_pos.npz').toarray()).to(device)
R_N_0 = torch.tensor(load_npz('rules/regu_neg.npz').toarray()).to(device)

' k=5 closure & k=2 closure'
R_P, R_N = closure(R_P_0, R_N_0, T=5, device=device)
R_P_2, R_N_2 = closure(R_P_0, R_N_0, T=2, device=device)

#R_P_3, R_N_3 = closure(R_P_0, R_N_0, T=3, device=device)
#R_P_k, R_N_k = closure(R_P_0, R_N_0, device=device)

#print(torch.count_nonzero(R_P_k - R_P))
#print(torch.count_nonzero(R_P - R_P_3))
#print(torch.count_nonzero(R_P_3 - R_P_2))
#
#print(torch.count_nonzero(R_N_k - R_N))
#print(torch.count_nonzero(R_N - R_N_3))
#print(torch.count_nonzero(R_N_3 - R_N_2))

print(f'max in R_P: {torch.max(R_P)}, R_N: {torch.max(R_N)}')

' weighted (by #pathways) closure, matmul as real mat '
' div by min(R) in matmul & ignore diff<5 paths '
R_weight = R_P - R_N
R_weight = torch.clamp(
        torch.mul(torch.sign(R_weight),
                  torch.max(torch.abs(R_weight)-5,
                            torch.zeros_like(R_weight))
                  ),-1,1)

' bool closure & simply diff (pos-neg) '
R_P, R_N = torch.clamp(R_P,0,1), torch.clamp(R_N,0,1)
R_P_2, R_N_2 = torch.clamp(R_P_2,0,1), torch.clamp(R_N_2,0,1)
R_diff = R_P - R_N
R_P, R_N = R_P.bool(), R_N.bool()

print(f'before closure: {int(torch.count_nonzero(torch.sum(R_P_0.bool() & R_N_0.bool(), dim=0)))} dual regulations, in total {int(torch.count_nonzero(torch.sum(R_P_0.bool() | R_N_0.bool(), dim=0)))}')

print(f'closure k=5: {int(torch.count_nonzero(torch.sum(R_P & R_N, dim=0)))} dual regulations, in total {int(torch.count_nonzero(torch.sum(R_P | R_N, dim=0)))}')

print(f'in weighted closure: {int(torch.sum(R_weight>0))} pos, {int(torch.sum(R_weight<0))} neg')

' combined closure, uses k=1 & k=2 at dual regulations '
R_comb = torch.where(R_P & R_N,
                     torch.where(R_P_2.bool()&R_N_2.bool(), 
                                 torch.clamp(R_P_0-R_N_0,-1,1),
                                 torch.clamp(R_P_2-R_N_2,-1,1)), R_diff)


X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy')).to(device).double()
Y_train = torch.tensor(np.load('dataset/precise1k/Y_train.npy'))
X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy')).to(device).double()
Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_train.npy'))


#NOTE tmp: load 623 gene indices
import pandas as pd
label_set = pd.read_csv('dataset/ncbi-sra/label_set.csv')
idx_list = list(label_set['matrix_idx'])

' in precise1k '
deduction_weight = (X_train @ R_weight).detach().cpu()[:,idx_list]
deduction_diff = (X_train @ R_diff).detach().cpu()[:,idx_list]
deduction_comb = (X_train @ R_comb).detach().cpu()[:,idx_list]

' mesure dual regulations only '
dual_idx = ((X_train @ R_P.double()).bool() & (X_train @ R_N.double()).bool()).detach().cpu()
dual_idx = dual_idx[:,idx_list]

print(f'\nmacro f1 on prec1k\n\
naive diff: {f1_score(torch.flatten(Y_train[dual_idx]), torch.flatten(deduction_diff[dual_idx]), average="macro")}\n\
weighted: {f1_score(torch.flatten(Y_train[dual_idx]), torch.flatten(deduction_weight[dual_idx]), average="macro")}\n\
combined: {f1_score(torch.flatten(Y_train[dual_idx]), torch.flatten(deduction_comb[dual_idx]), average="macro")}')

print(f'\nconfusion on prec1k\n\
naive diff:\n{confusion_matrix(torch.flatten(Y_train[dual_idx]), torch.flatten(deduction_diff[dual_idx]), labels=[-1, 0,1])}\n\
weighted:\n{confusion_matrix(torch.flatten(Y_train[dual_idx]), torch.flatten(deduction_weight[dual_idx]), labels=[-1, 0,1])}\n\
combined:\n{confusion_matrix(torch.flatten(Y_train[dual_idx]), torch.flatten(deduction_comb[dual_idx]), labels=[-1, 0,1])}')

' in ncbi-sra '
deduction_weight = (X_test @ R_weight).detach().cpu()[:,idx_list]
deduction_diff = (X_test @ R_diff).detach().cpu()[:,idx_list]
deduction_comb = (X_test @ R_comb).detach().cpu()[:,idx_list]

' mesure dual regulations only '
dual_idx = ((X_test @ R_P.double()).bool() & (X_test @ R_N.double()).bool()).detach().cpu()
dual_idx = dual_idx[:,idx_list]

print(f'\nmacro f1 on ncbi-sra\n\
naive diff: {f1_score(torch.flatten(Y_test[dual_idx]), torch.flatten(deduction_diff[dual_idx]), average="macro")}\n\
weighted: {f1_score(torch.flatten(Y_test[dual_idx]), torch.flatten(deduction_weight[dual_idx]), average="macro")}\n\
combined: {f1_score(torch.flatten(Y_test[dual_idx]), torch.flatten(deduction_comb[dual_idx]), average="macro")}')

print(f'\nconfusion on ncbi-sra\n\
naive diff:\n{confusion_matrix(torch.flatten(Y_test[dual_idx]), torch.flatten(deduction_diff[dual_idx]), labels=[-1, 0,1])}\n\
weighted:\n{confusion_matrix(torch.flatten(Y_test[dual_idx]), torch.flatten(deduction_weight[dual_idx]), labels=[-1, 0,1])}\n\
combined:\n{confusion_matrix(torch.flatten(Y_test[dual_idx]), torch.flatten(deduction_comb[dual_idx]), labels=[-1, 0,1])}')
