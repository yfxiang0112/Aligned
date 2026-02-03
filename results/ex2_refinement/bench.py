import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import torch
from sklearn.metrics import f1_score

from aligned.reasoner import RegulatoryKB

model_name = 'DCDI-G'
data_name = 'norman'
device = 'cuda'

df_bench = pd.read_csv(f'results/ex2_refinement/benchmark/{model_name}.csv', index_col=0)
df_ens = pd.read_csv('dataset/human/ensembl_mapping.csv')
df_genes = pd.read_csv('dataset/human/norman_gene_ann.csv', index_col=0)
ens_mapping = {row['gene_id']:row['gene_name'] for _,row in df_ens.iterrows()}

df_bench['source'] = df_bench['source'].apply(lambda x: ens_mapping[x])
df_bench['target'] = df_bench['target'].apply(lambda x: ens_mapping[x])

KB_bench_row = np.array(\
        [int(df_genes.loc[g,'vector_idx']) for g in df_bench['source']])
KB_bench_col = np.array(\
        [int(df_genes.loc[g,'vector_idx']) for g in df_bench['target']])
KB_bench_data = np.full_like(KB_bench_row, fill_value=1.)
KB_bench = coo_matrix((KB_bench_data, (KB_bench_row,KB_bench_col)), shape=(len(df_genes),len(df_genes)))
save_npz(f'results/ex2_refinement/benchmark/{model_name}.npz', KB_bench)

reasoner_bench = RegulatoryKB(
        pos_trn_pth=f'results/ex2_refinement/benchmark/{model_name}.npz',
        neg_trn_pth=None,
        device=device)
reasoner_bench.closure_(T=5, closure_type='naive')

print(f'{model_name} Inferred KB statics')
reasoner_bench.eval()

regu_p_pth = 'scripts/klg_refine/omnipath_P.npz'
regu_n_pth = 'scripts/klg_refine/omnipath_N.npz'
reasoner_true = RegulatoryKB(
        pos_trn_pth=regu_p_pth,
        neg_trn_pth=regu_n_pth,
        device=device)
reasoner_true.closure_(T=5, closure_type='naive')

print('\n\noriginal KB statics')
reasoner_true.eval()

n_cols = reasoner_true.KB.shape[1]
X = torch.eye(n_cols).to(device)
Y = reasoner_true.deduce(X)
Omega = torch.any((torch.clamp(X.T @ Y.float(), -1,1)!=0), axis=1)

true_R0_flat = reasoner_true.Regu_0.cpu().numpy().flatten()
true_R_flat = reasoner_true.KB[Omega].cpu().numpy().flatten()

bench_R0_flat = reasoner_bench.Regu_0.cpu().numpy().flatten()
bench_R_flat = reasoner_bench.KB[Omega].cpu().numpy().flatten()
    
f1_R0 = f1_score(np.abs(true_R0_flat), np.abs(bench_R0_flat), average='macro')
f1_R = f1_score((true_R_flat), np.abs(bench_R_flat), average='macro')
print(f'\nRk f1={f1_R:.4f}, R0 f1={f1_R0:.4f}')
