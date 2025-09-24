import pandas as pd
import numpy as np

    
labeled_input = pd.read_csv('dataset/ncbi-sra/metadata.csv')
gene_mapping = pd.read_csv('dataset/raw/genome_annotations.tsv', sep='\t')
gene_names = list(gene_mapping['Symbol'])

WT_idx_list = ['ERX1273486', 'ERX2874140', 'SRX4912066']
labeled_input = labeled_input[(labeled_input['overexpression']!='WT')|(labeled_input['accession'].isin(WT_idx_list))].reset_index()
data_col = labeled_input['overexpression']
idx_mask = list(labeled_input['index'])

''' input gene name to matrix '''
# Create an empty one-hot matrix
data_labeled = np.zeros((len(labeled_input), len(gene_names)), dtype=np.int8)

# keep the list of labeled gene index
labeled_genes = set()

# Populate the matrix
for i, value in enumerate(data_col):
    if value in gene_names:
        j = gene_names.index(value)  # Get index of the value in the categories list
        labeled_genes.add(j)
        data_labeled[i, j] = 1  # Set the corresponding position to 1
np.save('dataset/ncbi-sra/X_label.npy', data_labeled)
print('labeled', data_labeled.shape)
print(np.count_nonzero(data_labeled))

''' count to binary label '''
logfc = pd.read_csv('dataset/ncbi-sra/logfc.csv', index_col=0)
pvalue = pd.read_csv('dataset/ncbi-sra/pvalue.csv', index_col=0)
#logfc = logfc[logfc.index.isin(labeled_input.index)] # filter rows with no input
M_logfc = np.array(logfc)
M_pvalue = np.array(pvalue, dtype=np.float64)

''' decision array of p < .05 '''
decision_array = np.zeros_like(logfc, dtype=np.int8)
#for i in range(M_logfc.shape[1]):
decision_array[:, :] = np.where(
        (M_pvalue[:, :] < .05) ,
        np.where((M_logfc[:, :] >  1) , 1, 
        np.where((M_logfc[:, :] < -1) , -1, 0)), 0)
#decision_array[:, i] = np.where(
#        (M_logfc[:, i] < 1) ,
#        np.where((M_logfc[:, i] < -1) , -1, 0), 1)

decision_array = decision_array[idx_mask]
    
print('Y=\n',decision_array)
print('sum expr in all rows:\n', abs(decision_array).sum(axis=1))
np.save('dataset/ncbi-sra/Y_label.npy', decision_array)

print(f'Y shape: {decision_array.shape}, nonzero in Y: {np.count_nonzero(decision_array)}, {np.count_nonzero(decision_array) / (decision_array.shape[0]*decision_array.shape[1])} of all\n=1: {np.count_nonzero(decision_array == 1)}, =-1: {np.count_nonzero(decision_array == -1)}')
