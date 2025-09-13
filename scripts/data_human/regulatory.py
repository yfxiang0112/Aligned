import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz

data_name = 'norman'

go_thrs = .2 #NOTE .27 for GO only


df_go= pd.read_csv(f'rules/human/{data_name}_go.csv')
df_genes = pd.read_csv(f'dataset/human/{data_name}_gene_ann.csv', index_col=0)
df_go = df_go[(df_go['importance'] >= go_thrs) & (df_go['source']!=df_go['target'])]

#df_regu = pd.read_csv('rules/human/regulatory_dorothea.csv', index_col=0)
#df_regu.rename({'tf':'source'}, axis=1, inplace=True)
#df_tf_ann = pd.read_csv('rules/human/tf_ann_dorothea.csv', index_col=0)

df_regu = pd.read_csv('rules/human/omnipath.csv', index_col=0)

#df_regu_p = df_regu[\
#        (df_regu['source'].isin(df_tf_ann.loc[df_tf_ann['class']!='repressors', 'tf']))\
#        & (df_regu['source'].isin(df_genes.index)) & (df_regu['target'].isin(df_genes.index))]
df_regu_p = df_regu[\
        (df_regu['is_stimulation'] == True)\
        & (df_regu['source'].isin(df_genes.index))\
        & (df_regu['target'].isin(df_genes.index))]
#df_regu_n = df_regu[\
#        (df_regu['source'].isin(df_tf_ann.loc[df_tf_ann['class']!='activators', 'tf']))\
#        & (df_regu['source'].isin(df_genes.index)) & (df_regu['target'].isin(df_genes.index))]
df_regu_n = df_regu[\
        (df_regu['is_inhibition'] == True)\
        & (df_regu['source'].isin(df_genes.index))\
        & (df_regu['target'].isin(df_genes.index))]

KB_P_row = np.array(\
        [int(df_genes.loc[g,'vector_idx']) for g in df_regu_p['source']]\
        + [int(df_genes.loc[g,'vector_idx']) for g in df_go['source']])
KB_P_col = np.array(\
        [int(df_genes.loc[g,'vector_idx']) for g in df_regu_p['target']]\
        + [int(df_genes.loc[g,'vector_idx']) for g in df_go['target']])
KB_P_data = np.full_like(KB_P_row, fill_value=1.)
KB_P = coo_matrix((KB_P_data, (KB_P_row,KB_P_col)), shape=(len(df_genes),len(df_genes)))
save_npz(f'rules/human/{data_name}_KB_P.npz', KB_P)

KB_N_row = np.array(\
        [int(df_genes.loc[g,'vector_idx']) for g in df_regu_n['source']])
        #+ [int(df_genes.loc[g,'vector_idx']) for g in df_go['source']])
KB_N_col = np.array(\
        [int(df_genes.loc[g,'vector_idx']) for g in df_regu_n['target']])
        #+ [int(df_genes.loc[g,'vector_idx']) for g in df_go['target']])
KB_N_data = np.full_like(KB_N_row, fill_value=1.)
KB_N = coo_matrix((KB_N_data, (KB_N_row,KB_N_col)), shape=(len(df_genes),len(df_genes)))
save_npz(f'rules/human/{data_name}_KB_N.npz', KB_N)

KB_P, KB_N = KB_P.toarray(), KB_N.toarray()
print(f'shape {KB_P.shape}')
print(f'#edges: pos {np.count_nonzero(KB_P)}, neg {np.count_nonzero(KB_N)}')
print(f'#out nodes: pos {np.count_nonzero(np.sum(KB_P,axis=1))}, neg {np.count_nonzero(np.sum(KB_N,axis=1))}')
print(f'>5% targets: pos {np.count_nonzero(np.sum(KB_P,axis=1)>(.05*KB_P.shape[1]))}, neg {np.count_nonzero(np.sum(KB_N,axis=1)>(.05*KB_N.shape[1]))}')
