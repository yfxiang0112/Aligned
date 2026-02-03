import pandas as pd
from aligned.reasoner import RegulatoryKB

gene_ann = pd.read_csv('dataset/human/norman_gene_ann.csv',index_col='gene_name')
KB = RegulatoryKB('rules/human/norman_KB_P.npz','rules/human/norman_KB_N.npz',device='cpu')

edges = pd.read_csv('data_anal/xref_csbench/recovery_KB/norman_interactions_66.csv',index_col=0)
KB.load('models/GNN_norman_KB_recovery_2_ABL_0.npz')

i = list(edges.source.apply(lambda x: gene_ann['vector_idx'][x]))
j = list(edges.target.apply(lambda x: gene_ann['vector_idx'][x]))
edges.insert(3, 'recovered', [int(i) for i in KB.Regu_0[i,j]])

edges.insert(2, 'significant_in_data', edges['data_corr'].apply(lambda x: x>1))
edges.insert(3, 'is_recovered', edges['recovered'].apply(lambda x: x!=0))

edges.sort_values('significant_in_data', ascending=False, inplace=True)
print(edges.iloc[:,:4])
