import pandas as pd
from tqdm import tqdm
import json
import pygraphblas as pgb

#NOTE: not used for tmp
#''' read goa df '''
#goa_df = pd.read_csv('rules/raw_goa/goa_mapping.csv', header=None)
#print(goa_df)
#
#''' list all gene ids and sort '''
#gene_ids = set()
#for gene_lst in goa_df[1]:
#    for g in eval(gene_lst):
#        gene_ids.add(g)
#gene_ids = list(gene_ids)
#gene_ids.sort()
#
#''' construct dict '''
#goa_inv = {}
#for idx, g in enumerate(tqdm(gene_ids)):
#    goa_inv[g] = set()
#    for _, row in goa_df.iterrows():
#        if g in eval(row[1]):
#            goa_inv[g].add('http://purl.obolibrary.org/obo/'+row[0])
#    goa_inv[g] = list(goa_inv[g])


''' save json '''
#with open('rules/annotations.json', 'w') as f:
#    json.dump(goa_inv, f, indent=4)
with open('rules/annotations.json', 'r') as f:
    goa_inv = json.load(f)

''' save matrix '''
with open('rules/node_idx.json', 'r') as f:
    dict_node2idx = json.load(f)

gene_list = pd.read_csv('dataset/gene_list.csv', index_col=0).transpose()
num_nodes = max(dict_node2idx.values()) + 1
num_genes = len(gene_list)
matrix = pgb.Matrix.sparse(pgb.types.BOOL, num_genes, num_nodes)

for gene_idx, gene in enumerate(gene_list['Geneid']):
    if gene not in goa_inv.keys():
        continue
    term_list = goa_inv[gene]
    for term in term_list:
        if term in dict_node2idx:
            node_idx = dict_node2idx[term]
            matrix[gene_idx, node_idx] = True

matrix.to_binfile('rules/annotations_bin')
