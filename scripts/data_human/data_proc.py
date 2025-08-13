import pickle
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz

def diff_expr(x0, x1):
    res = np.where(x1 >  x0, 1,
                   np.where(x1 <  x0, -1, 0))
    return res

gene2go = pickle.load(open('../GEARS/gene2go_all.pkl', 'rb'))

for data_name in ['norman', 'dixit', 'adamson']:
    data_dict = pickle.load(open(f'../GEARS/{data_name}/data_pyg/cell_graphs.pkl','rb'))
    ann_data = sc.read_h5ad(f'../GEARS/{data_name}/perturb_processed.h5ad')
    #df_go= pd.read_csv(f'../GEARS/{data_name}/go.csv')
    df_genes= pd.DataFrame(ann_data.var)
    df_genes['vector_idx'] = list(range(len(df_genes)))
    df_genes.set_index('gene_name', inplace=True)
    df_genes.to_csv(f'dataset/human/{data_name}_gene_ann.csv')


    Y = []
    X_row, X_col = [], []
    metadata = []
    data = [data_dict[k] for k in data_dict.keys()]

    genes_out_key = set()

    for i, lst in enumerate(data):
        pert_idx = [int(i) for i in lst[0].pert_idx]
        pert = lst[0].pert.split('+')
        de_idx = [int(i) for i in lst[0].de_idx]
    
        start_data_idx = len(Y)
        Y += [diff_expr(np.squeeze(d.x, -1), np.squeeze(d.y, 0)) for d in lst]
        end_data_idx = len(Y)
    
        metadata.append({'pert':pert, 'pert_idx':pert_idx, 'data_start_idx':start_data_idx, 'data_end_idx+1':end_data_idx, 'de_idx':de_idx})

        for g in pert:
            if g == 'ctrl':
                continue
            elif g in df_genes.index:
                for row in range(start_data_idx, end_data_idx):
                    X_row.append(row)
                    X_col.append(int(df_genes.loc[g,'vector_idx']))
            else:
                genes_out_key.add(g)
    
    Y = np.stack(Y)
    print(len(Y))
    print(Y)
    #np.save(f'dataset/human/{data_name}_Y.npy', Y)
    save_npz(f'dataset/human/{data_name}_Y.npz', coo_matrix(Y, shape=Y.shape))

    X_data = np.array([1.]*len(X_row))
    X_row, X_col = np.array(X_row), np.array(X_col)
    X = coo_matrix((X_data, (X_row,X_col)), shape=Y.shape)
    save_npz(f'dataset/human/{data_name}_X.npz', X)

    #KB_row = np.array([int(df_genes.loc[g,'vector_idx']) for g in df_go['source']])
    #KB_col = np.array([int(df_genes.loc[g,'vector_idx']) for g in df_go['target']])
    #KB_data = np.full_like(KB_row, fill_value=1.)
    #KB = coo_matrix((KB_data, (KB_row,KB_col)), shape=(len(df_genes),len(df_genes)))
    #save_npz(f'dataset/human/{data_name}_KB.npz', KB)
    
    metadata = pd.DataFrame(metadata)
    print(metadata)
    metadata.to_csv(f'dataset/human/{data_name}_metadata.csv')

    print(f'{len(genes_out_key)} pert genes not in dataset')
