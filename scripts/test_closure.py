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
    
        R_P, R_N = torch.clamp(R_P,0,1), torch.clamp(R_N,0,1)
        #R_P /= torch.min(R_P[R_P!=0])
        #R_N /= torch.min(R_N[R_N!=0])
    
        if torch.all(R_P == R_P_) and torch.all(R_N == R_N_):
            break
        cnt += 1
    
    #R_N = torch.where(I.bool(), 0, R_N)
    return R_P, R_N

device = 'cuda'
R_P_0 = torch.tensor(load_npz('rules/regu_pos.npz').toarray()).to(device)
R_N_0 = torch.tensor(load_npz('rules/regu_neg.npz').toarray()).to(device)

R_P, R_N = closure(R_P_0, R_N_0, T=5, device=device)

R_C_0 = torch.clamp(torch.abs(R_P_0)+torch.abs(R_N_0), 0,1)
R_C = torch.matrix_power(R_C_0, 4)
R_P_C = torch.clamp(R_P_0 @ R_C + R_C @ R_P_0, 0,1)
R_N_C = torch.clamp(R_N_0 @ R_C + R_C @ R_N_0, 0,1)

R_C_5 = torch.clamp(torch.matrix_power(R_C_0, 5), 0,1)

total = R_P.shape[0] * R_P.shape[1]
print(torch.count_nonzero(R_P != R_P_C) / total)
print(torch.count_nonzero(R_N != R_N_C) / total)
print(torch.count_nonzero(R_C_5 != R_P) / total)
print(torch.count_nonzero(R_C_5 != R_N) / total)

print(torch.count_nonzero(R_C_5 != torch.clamp(R_P + R_N, 0,1)) / total)

print(torch.count_nonzero(R_P_0 != R_P) / total)
