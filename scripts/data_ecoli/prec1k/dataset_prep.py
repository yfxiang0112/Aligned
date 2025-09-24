import pandas as pd
import numpy as np
    
df_metadata = pd.read_csv('dataset/precise1k/metadata.csv', index_col=0)
data_index = df_metadata.index

gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
idx_dict = {locus:idx for idx,locus in enumerate(gene_idx['locus'])}


''' input gene name to matrix '''
input_labeled = df_metadata['perturbation']
x_labeled = np.zeros((len(df_metadata), len(gene_idx)), dtype=np.int8)

# keep the list of labeled gene index
labeled_genes = set()

# Populate the matrix
for i, value in enumerate(df_metadata['perturbation']):
    for g,pert in eval(value).items():
        j = idx_dict[g]
        labeled_genes.add(j)
        x_labeled[i, j] = pert
print(len(x_labeled))
np.save('dataset/precise1k/X_label.npy', x_labeled)
print('labeled', x_labeled.shape)
print('nonzero: ', np.count_nonzero(x_labeled))

''' count to ternary label '''
logfc = pd.read_csv('dataset/precise1k/logfc.csv', index_col=0)
pvalue = pd.read_csv('dataset/precise1k/pvalue.csv', index_col=0)
logfc, pvalue = logfc.loc[:,logfc.columns.isin(set(gene_idx['locus']))], pvalue.loc[:,logfc.columns.isin(set(gene_idx['locus']))]
print(logfc.shape)
logfc, pvalue = logfc.loc[data_index], pvalue.loc[data_index]
M_logfc = np.array(logfc)
M_pvalue = np.array(pvalue, dtype=np.float64)

''' decision array of p < .05 '''
decision_array = np.zeros_like(logfc, dtype=np.int8)

decision_array[:, :] = np.where(
        (M_pvalue[:, :] < .05) ,
        np.where((M_logfc[:, :] >  1) , 1, 
        np.where((M_logfc[:, :] < -1) , -1, 0)), 0)
#decision_array[:, i] = np.where(
#        (M_logfc[:, i] < 1) ,
#        np.where((M_logfc[:, i] < -1) , -1, 0), 1)

print('Y=\n',decision_array)
print('sum expr in all rows:\n', abs(decision_array).sum(axis=1))
np.save('dataset/precise1k/Y_label.npy', decision_array)

print(f'Y shape: {decision_array.shape}, nonzero in Y: {np.count_nonzero(decision_array)}, {np.count_nonzero(decision_array) / (decision_array.shape[0]*decision_array.shape[1])} of all\n=1: {np.count_nonzero(decision_array == 1)}, =-1: {np.count_nonzero(decision_array == -1)}')


label_set = pd.read_csv('dataset/ncbi-sra/label_set.csv')
idx_list = list(gene_idx.loc[list(label_set['matrix_idx']), 'precise1k_idx'])
np.save('dataset/precise1k/Y_train.npy', decision_array[:,idx_list])
