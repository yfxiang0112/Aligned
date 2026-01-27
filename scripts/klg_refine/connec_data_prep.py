from cupy import load
import numpy as np
from scipy.sparse import load_npz
import pandas as pd

pos_regu = load_npz('rules/regu_pos.npz').toarray()
neg_regu = load_npz('rules/regu_neg.npz').toarray()
connectv_kb = np.clip(np.abs(pos_regu) + np.abs(neg_regu), 0,1)

''' mat mul X (pert) -> Y & abs: get connectivity matrix
    (as supervision for mat compl) '''
X_l = np.load('dataset/precise1k/X_label.npy')
Y_l = np.load('dataset/precise1k/Y_label.npy')
connectv_sup = np.clip(np.abs(X_l).T @ np.abs(Y_l), 0,1)

''' align with precise1k genome '''
gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
idx_map = [idx for idx,v in enumerate(gene_idx['precise1k_idx']) if v != -1]
connectv_kb = connectv_kb[idx_map][:,idx_map]
connectv_sup = connectv_sup[idx_map][:,[i for i in gene_idx['precise1k_idx'] if i != -1]]

print(connectv_kb.shape, connectv_sup.shape)
