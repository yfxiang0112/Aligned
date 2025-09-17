import pandas as pd
import gseapy
#from neteval import run_network_evaluation
import networkx as nx
import numpy as np
import json


data_name = 'norman'
ann = pd.read_csv(f'dataset/human/{data_name}_gene_ann.csv')
genes = list(ann['gene_name'])

#KB = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
#                  neg_trn_pth='rules/human/{data_name}_KB_N.npz',
#                  device='cpu')

enr = gseapy.enrichr(
    gene_list=genes,
    gene_sets=['KEGG_2021_Human'],) # or 'Reactome_2022'

print(enr.results[['Term', 'Overlap', 'Adjusted P-value']])

gene_sets = {}
for _, row in enr.results.iterrows():
    pathway = row['Term']
    # Overlap column looks like "3/56" → number of genes found
    overlapping_genes = set(row['Genes'].split(";"))
    gene_sets[pathway] = str(overlapping_genes)

print("Example gene sets for netteval:")
for k, v in list(gene_sets.items())[:3]:
    print(k, ":", v)

json.dump(gene_sets, open(f'scripts/net_eval/{data_name}_gene_set.json', 'w'), indent=4)


#adj = torch.clamp(torch.abs(KB.Regu_P_0) + torch.abs(KB.Regu_N_0), 0,1).numpy()
#G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
#G = nx.relabel_nodes(G, dict(enumerate(genes)))
#
## Run gene set recovery
#results = run_gene_set_recovery(G, gene_sets, n_iter=100)
#print(results)
#
##################
#
#KB.load('models/GNN_norman_Sep15_2_ABL_0.npz')
#
#adj = torch.clamp(torch.abs(KB.Regu_P_0) + torch.abs(KB.Regu_N_0), 0,1).numpy()
#G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
#G = nx.relabel_nodes(G, dict(enumerate(genes)))
#
## Run gene set recovery
#results = run_gene_set_recovery(G, gene_sets, n_iter=100)
#print(results)
