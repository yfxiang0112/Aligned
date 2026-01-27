import numpy as np
from scipy.sparse import coo_matrix, save_npz
import pandas as pd

gene_annotations = pd.read_csv('dataset/raw/genome_annotations.tsv', sep='\t')
gene_names = gene_annotations['Symbol']
node_idx = {}
for idx, gene in enumerate(gene_names):
    node_idx[gene] = idx
num_nodes = len(gene_annotations)

trn_df = pd.read_csv('rules/regulatory_ecocyc.csv')

empty_gene_cnt = {}

row_pos, col_pos, row_neg, col_neg = [],[],[],[]
shape = (num_nodes, num_nodes)

for idx, row in trn_df.iterrows():
    out_node = row['regulator']
    in_node = row['regulated']
    
    if out_node not in node_idx:
        if out_node not in empty_gene_cnt:
            empty_gene_cnt[out_node] = 1
        else:
            empty_gene_cnt[out_node] += 1
        continue
    if in_node not in node_idx:
        if in_node not in empty_gene_cnt:
            empty_gene_cnt[in_node] = 1
        else:
            empty_gene_cnt[in_node] += 1
        continue

    if row['edge'] == '+':
        row_pos.append(node_idx[out_node])
        col_pos.append(node_idx[in_node])
    elif row['edge'] == '-':
        row_neg.append(node_idx[out_node])
        col_neg.append(node_idx[in_node])
    elif row['edge'] == '-+':
        row_pos.append(node_idx[out_node])
        col_pos.append(node_idx[in_node])

        row_neg.append(node_idx[out_node])
        col_neg.append(node_idx[in_node])

pos_regu = coo_matrix(([1.]*len(row_pos), (row_pos,col_pos)), shape)
neg_regu = coo_matrix(([1.]*len(row_neg), (row_neg,col_neg)), shape)

save_npz('rules/regu_pos.npz', pos_regu)
save_npz('rules/regu_neg.npz', neg_regu)

print(empty_gene_cnt)
