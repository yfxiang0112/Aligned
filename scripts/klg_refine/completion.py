import torch
import numpy as np
from scipy.sparse import load_npz, coo_matrix, save_npz
from sklearn.metrics import f1_score
import gc

from egoal.reasoner import RegulatoryKB

seed = 42
model_name = 'mix_1'
device = 'cuda:5'

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

regu_p_pth = 'scripts/klg_refine/omnipath_P.npz'
regu_n_pth = 'scripts/klg_refine/omnipath_N.npz'
#regu_p_pth = 'rules/regu_pos.npz'
#regu_n_pth = 'rules/regu_neg.npz'
incomp_p_pth = 'scripts/klg_refine/kb/incomp_P.npz'
incomp_n_pth = 'scripts/klg_refine/kb/incomp_N.npz'

reasoner_true = RegulatoryKB(
        pos_trn_pth=regu_p_pth,
        neg_trn_pth=regu_n_pth,
        device=device)
reasoner_true.closure_(T=5, closure_type='naive')

print('original KB statics')
reasoner_true.eval()

n_cols = reasoner_true.KB.shape[1]
X = torch.eye(n_cols).to(device)
Y = reasoner_true.deduce(X)

' prepare ground-truth kb '
R_P = load_npz(regu_p_pth).toarray()
R_N = load_npz(regu_n_pth).toarray()
R = np.clip(R_P+R_N, 0,1)

for p_incompl in [0., .05, .1, .2, .3, .4, .5, .7, .9]:

    ' create mask for p% nonzero positions '
    nonzero_indices = np.argwhere(R != 0)
    zero_indices = np.argwhere(R == 0)
    num_nonzero = len(nonzero_indices)
    if num_nonzero == 0:
        mask = np.zeros_like(R, dtype=bool)
    num_to_select = int(p_incompl * num_nonzero)

    remove_indices = np.random.choice(num_nonzero, size=int(num_to_select/2), replace=False)
    remove_mask = np.zeros_like(R, dtype=bool)
    remove_positions = nonzero_indices[remove_indices]
    for row, col in remove_positions:
        remove_mask[row, col] = True

    positive_indices = np.random.choice(num_nonzero, size=int(num_to_select/4), replace=False)
    positive_mask = np.zeros_like(R, dtype=bool)
    positive_positions = zero_indices[positive_indices]
    for row, col in positive_positions:
        positive_mask[row, col] = True

    negative_indices = np.random.choice(num_nonzero, size=int(num_to_select/4), replace=False)
    negative_mask = np.zeros_like(R, dtype=bool)
    negative_positions = zero_indices[negative_indices]
    for row, col in negative_positions:
        negative_mask[row, col] = True


    ' mask & save incomplete KB '
    save_npz(incomp_p_pth, coo_matrix(np.where(remove_mask, 0, 
        np.where(positive_mask, 1, R_P))))
    save_npz(incomp_n_pth, coo_matrix(np.where(remove_mask, 0, 
        np.where(negative_mask, 1, R_N))))


    reasoner_train = RegulatoryKB(
            pos_trn_pth= incomp_p_pth,
            neg_trn_pth= incomp_n_pth,
            device=device)
    reasoner_train.refine(X= X,
                    Y= Y,
                    k= 5,
                    t= 1,
                    t0= 100,
                    epochs= 3000,
                    lr= 1e-3,
                    verbose= False)
    reasoner_train.save(f'scripts/klg_refine/kb/restored_{p_incompl}_{model_name}.npz')

    Omega = torch.any((torch.clamp(X.T @ Y.float(), -1,1)!=0), axis=1)

    true_R0P_flat, true_R0N_flat = R_P.flatten(), R_N.flatten()
    true_R0_flat = reasoner_true.Regu_0.cpu().numpy().flatten()
    
    true_RP_flat = reasoner_true.KB_P[Omega].cpu().numpy().flatten()
    true_RN_flat = reasoner_true.KB_N[Omega].cpu().numpy().flatten()
    true_R_flat = reasoner_true.KB[Omega].cpu().numpy().flatten()


    pred_R0P_flat = reasoner_train.Regu_P_0.cpu().numpy().flatten()
    pred_R0N_flat = reasoner_train.Regu_N_0.cpu().numpy().flatten()
    pred_R0_flat = reasoner_train.Regu_0.cpu().numpy().flatten()

    pred_RP_flat = reasoner_train.KB_P[Omega].cpu().numpy().flatten()
    pred_RN_flat = reasoner_train.KB_N[Omega].cpu().numpy().flatten()
    pred_R_flat = reasoner_train.KB[Omega].cpu().numpy().flatten()

    f1_R0P = f1_score(true_R0P_flat, pred_R0P_flat, average='macro')
    f1_R0N = f1_score(true_R0N_flat, pred_R0N_flat, average='macro')
    f1_R0 = f1_score(true_R0_flat, pred_R0_flat, average='macro')
    idx_nonzero = (true_R0_flat != 0) | (pred_R0_flat != 0)
    acc_R0 = np.sum(true_R0_flat[idx_nonzero] == pred_R0_flat[idx_nonzero]) / len(true_R0_flat[idx_nonzero])

    f1_RP = f1_score(true_RP_flat, pred_RP_flat, average='macro')
    f1_RN = f1_score(true_RN_flat, pred_RN_flat, average='macro')
    f1_R = f1_score(true_R_flat, pred_R_flat, average='macro')
    idx_nonzero = (true_R_flat != 0) | (pred_R_flat != 0)
    acc_R = np.sum(true_R_flat[idx_nonzero] == pred_R_flat[idx_nonzero]) / len(true_R_flat[idx_nonzero])

    print(f'--- KB recovery: incompleteness p = {p_incompl} ---')
    print(f'F1 on initial KB, pos: {f1_R0P: .5f}, neg: {f1_R0N: .5f}, combined: {f1_R0: .5f}, acc: {acc_R0: .5f}')
    print(f'F1 on closure KB, pos: {f1_RP: .5f}, neg: {f1_RN: .5f}, combined: {f1_R: .5f}, acc: {acc_R: .5f}')
    print('KB statics:')
    reasoner_train.eval()
    print('----------\n')

    del reasoner_train
    gc.collect()
    torch.cuda.empty_cache()
