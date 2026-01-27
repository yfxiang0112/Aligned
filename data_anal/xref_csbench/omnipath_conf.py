import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz

data_name = 'norman'

df_regu = pd.read_csv('rules/human/omnipath.csv', index_col=0)
df_genes = pd.read_csv(f'dataset/human/{data_name}_gene_ann.csv', index_col=0)

df_regu = df_regu[\
          (df_regu['source'].isin(df_genes.index))\
        & (df_regu['target'].isin(df_genes.index))]

row = np.array( [int(df_genes.loc[g,'vector_idx']) for g in df_regu['source']])
col = np.array( [int(df_genes.loc[g,'vector_idx']) for g in df_regu['target']])
data = np.array(df_regu['n_references']) +\
        np.array(df_regu['curation_effort'])
print(data)

KB_conf = coo_matrix((data, (row,col)), shape=(len(df_genes),len(df_genes)))
save_npz(f'data_anal/xref_csbench/networks/omni_conf.npz', KB_conf)
