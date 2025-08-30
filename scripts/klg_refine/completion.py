import torch
import numpy as np
from scipy.sparse import load_npz, coo_matrix, save_npz
from sklearn.metrics import f1_score
import gc

from egoal.reasoner import RegulatoryKB

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

regu_p_pth = 'rules/human/norman_KB_P.npz'
regu_n_pth = 'rules/human/norman_KB_N.npz'
#regu_p_pth = 'rules/regu_pos.npz'
#regu_n_pth = 'rules/regu_neg.npz'
incomp_p_pth = 'scripts/klg_refine/KB_P.npz'
incomp_n_pth = 'scripts/klg_refine/KB_N.npz'

device = 'cuda:6'

reasoner_true = RegulatoryKB(
        pos_trn_pth=regu_p_pth,
        neg_trn_pth=regu_n_pth,
        device=device)
reasoner_true.closure_(T=5, closure_type='naive')

n_cols = reasoner_true.KB.shape[1]
X = torch.eye(n_cols).to(device)
Y = reasoner_true.deduce(X)

' prepare ground-truth kb '
R_P = load_npz(regu_p_pth).toarray()
R_N = load_npz(regu_n_pth).toarray()
R = np.clip(R_P+R_N, 0,1)

true_R0P_flat, true_R0N_flat = R_P.flatten(), R_N.flatten()
true_R0_flat = reasoner_true.Regu_0.cpu().numpy().flatten()

true_RP_flat = reasoner_true.KB_P.cpu().numpy().flatten()
true_RN_flat = reasoner_true.KB_N.cpu().numpy().flatten()
true_R_flat = reasoner_true.KB.cpu().numpy().flatten()

for p_incompl in [0., .05, .1, .2, .3, .4, .5]:

    ' create mask for p% nonzero positions '
    nonzero_indices = np.argwhere(R != 0)
    num_nonzero = len(nonzero_indices)
    if num_nonzero == 0:
        mask = np.zeros_like(R, dtype=bool)

    num_to_select = max(1, int(p_incompl * num_nonzero))
    selected_indices = np.random.choice(num_nonzero, size=num_to_select, replace=False)
    mask = np.zeros_like(R, dtype=bool)
    selected_positions = nonzero_indices[selected_indices]
    for row, col in selected_positions:
        mask[row, col] = True

    ' mask & save incomplete KB '
    save_npz(incomp_p_pth, coo_matrix(np.where(mask, 0, R_P)))
    save_npz(incomp_n_pth, coo_matrix(np.where(mask, 0, R_N)))


    reasoner_train = RegulatoryKB(
            pos_trn_pth= incomp_p_pth,
            neg_trn_pth= incomp_n_pth,
            device=device)
    reasoner_train.refine(X= X,
                    Y= Y,
                    k= 5,
                    epochs= 5000,
                    init_lr= 1e-3,
                    verbose= True)


    pred_R0P_flat = reasoner_train.Regu_P_0.cpu().numpy().flatten()
    pred_R0N_flat = reasoner_train.Regu_N_0.cpu().numpy().flatten()
    pred_R0_flat = reasoner_train.Regu_0.cpu().numpy().flatten()

    pred_RP_flat = reasoner_train.KB_P.cpu().numpy().flatten()
    pred_RN_flat = reasoner_train.KB_N.cpu().numpy().flatten()
    pred_R_flat = reasoner_train.KB.cpu().numpy().flatten()

    f1_R0P = f1_score(true_R0P_flat, pred_R0P_flat, average='macro')
    f1_R0N = f1_score(true_R0N_flat, pred_R0N_flat, average='macro')
    f1_R0 = f1_score(true_R0_flat, pred_R0_flat, average='macro')

    f1_RP = f1_score(true_RP_flat, pred_RP_flat, average='macro')
    f1_RN = f1_score(true_RN_flat, pred_RN_flat, average='macro')
    f1_R = f1_score(true_R_flat, pred_R_flat, average='macro')

    print(f'--- KB recovery: incompleteness p = {p_incompl} ---')
    print(f'F1 on initial KB, pos: {f1_R0P: .5f}, neg: {f1_R0N: .5f}, combined: {f1_R0: .5f}')
    print(f'F1 on closure KB, pos: {f1_RP: .5f}, neg: {f1_RN: .5f}, combined: {f1_R: .5f}')
    print('----------\n')

    del reasoner_train
    gc.collect()
    torch.cuda.empty_cache()
