import networkx as nx
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
import copy
import torch
import json
import pandas as pd
import gseapy
from tqdm import tqdm

from egoal.reasoner import RegulatoryKB

def network_diffusion_scores(G, seed_genes, alpha=0.85, tol=1e-6, max_iter=1000):
    """
    Do a simple random‐walk with restart / diffusion from seed_genes.
    Returns a dictionary: gene -> diffusion score.
    """
    # initialize
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    # adjacency matrix row-normalized
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, format='csr')
    # Row normalize
    row_sums = np.array(A.sum(axis=1)).flatten()
    # avoid division by zero
    row_idx, col_idx = A.nonzero()
    # normalize each row
    for i, j in zip(row_idx, col_idx):
        if row_sums[i] > 0:
            A[i, j] /= row_sums[i]
    # initialize score vector
    # personalization for RWR: seed_genes get uniform weight
    p0 = np.zeros(n)
    for s in seed_genes:
        if s in node_index:
            p0[node_index[s]] = 1.0
    if len(seed_genes) > 0:
        p0 = p0 / np.sum(p0)
    else:
        # nothing to propagate
        return {node:0.0 for node in nodes}

    p = p0.copy()
    teleport = p0.copy()

    for iteration in range(max_iter):
        p_new = alpha * (A.transpose().dot(p)) + (1 - alpha) * teleport
        if np.linalg.norm(p_new - p, ord=1) < tol:
            break
        p = p_new

    scores = {node: p[node_index[node]] for node in nodes}
    return scores

def degree_matched_shuffle(G, seed_genes, num_shuffles=1000):
    """
    Produce null scores by randomly selecting seed gene sets matched for degree distribution.
    Returns list of score dicts (one per shuffle).
    """
    degrees = dict(G.degree())
    # group nodes by degree
    deg_to_nodes = defaultdict(list)
    for node, d in degrees.items():
        deg_to_nodes[d].append(node)

    shuffled_scores = []
    seed_deg = [degrees[s] for s in seed_genes if s in degrees]
    # for each shuffle, pick random nodes of same degrees
    nodes = list(G.nodes())
    for _ in range(num_shuffles):
        shuffle_set = []
        for d in seed_deg:
            candidates = deg_to_nodes.get(d, [])
            if not candidates:
                # fallback: any node
                shuffle_set.append(random.choice(nodes))
            else:
                shuffle_set.append(random.choice(candidates))
        # compute diffusion scores from this shuffle
        sc = network_diffusion_scores(G, shuffle_set)
        shuffled_scores.append(sc)
    return shuffled_scores

def gene_set_recovery(G, gene_sets, all_genes=None, shuffle_num=100, alpha=0.85):
    """
    Perform gene set recovery for each gene set using diffusion scoring.
    G: networkx Graph
    gene_sets: dict {set_name: set_of_genes}
    all_genes: optional set/list of all genes to consider
    shuffle_num: number of null shuffles
    alpha: diffusion parameter
    Returns: dict {set_name: { 'AUROC': , 'AUPRC': }}
    """
    if all_genes is None:
        all_genes = set(G.nodes())
    else:
        all_genes = set(all_genes)

    results = {}

    for setname, genes in tqdm(gene_sets.items(), total=len(gene_sets)):
        genes_in_net = list(set(genes) & set(G.nodes()))
        if len(genes_in_net) < 2:
            # ignore tiny sets
            continue

        # Compute true diffusion scores: seed = genes_in_net
        true_scores = network_diffusion_scores(G, genes_in_net, alpha=alpha)

        # Define ground truth: whether each gene is in the gene set
        y_true = []
        y_score = []
        for node in all_genes:
            y_true.append(1 if node in genes_in_net else 0)
            y_score.append(true_scores.get(node, 0.0))

        # Compute metrics on the true scores
        auroc_true = roc_auc_score(y_true, y_score)
        auprc_true = average_precision_score(y_true, y_score)

        # Null: shuffle seed sets to get null score distributions
        #null_aurocs = []
        #null_auprcs = []
        #shuffles = degree_matched_shuffle(G, genes_in_net, num_shuffles=shuffle_num)
        #for sc in shuffles:
        #    y_score_sh = [sc.get(node, 0.0) for node in all_genes]
        #    try:
        #        auroc_sh = roc_auc_score(y_true, y_score_sh)
        #        auprc_sh = average_precision_score(y_true, y_score_sh)
        #    except ValueError:
        #        # possibly all y_true are 0 or all 1; skip
        #        continue
        #    null_aurocs.append(auroc_sh)
        #    null_auprcs.append(auprc_sh)

        results[setname] = {
            'AUROC_true': auroc_true,
            'AUPRC_true': auprc_true,
            #'null_AUROC': null_aurocs,
            #'null_AUPRC': null_auprcs,
            # you might also compute empirical p-value, z-score etc.
        }

    return results

def get_gene_sets(gene_list,
                  database_lst= ['KEGG_2021_Human']):
    enr = gseapy.enrichr(
        gene_list=gene_list,
        gene_sets=database_lst,) # or 'Reactome_2022'

    gene_sets = {}
    for _, row in enr.results.iterrows():
        pathway = row['Term']
        # Overlap column looks like "3/56" → number of genes found
        overlapping_genes = set(row['Genes'].split(";"))
        gene_sets[pathway] = overlapping_genes

    return gene_sets

# Example usage:

if __name__ == "__main__":
    # Load or build your network G
    data_name = 'norman'
    model_name = 'GNN_norman_Sep15_1_ABL_0'
    load_model_pth = f'scripts/net_eval/models/{model_name}.npz'
    database = 'kegg'

    database_lst = ['Reactome_2022' if database=='reactome' else 'KEGG_2021_Human']
    ann = pd.read_csv(f'dataset/human/{data_name}_gene_ann.csv')
    genes = list(ann['gene_name'])
    
    KB = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device='cpu')
    if load_model_pth != None:
        KB.load(load_model_pth)

    adj = torch.clamp(torch.abs(KB.Regu_P_0) + torch.abs(KB.Regu_N_0), 0,1).numpy()
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, dict(enumerate(genes)))

    gene_sets = get_gene_sets(genes,
                              database_lst = database_lst)
    print('--- collected gene sets ---')


    res = gene_set_recovery(G, gene_sets, all_genes=genes, shuffle_num=200, alpha=0.85)

    for setname, metrics in res.items():
        print(f"Gene set: {setname}")
        print("  AUROC_true:", metrics['AUROC_true'])
        print("  AUPRC_true:", metrics['AUPRC_true'])
        # Compute p‐value: fraction of null ≥ true
        #p_auroc = sum(1 for x in metrics['null_AUROC'] if x >= metrics['AUROC_true']) / len(metrics['null_AUROC'])
        #print("  p-value (AUROC):", p_auroc)
        print()

    json.dump(res, open(f'scripts/net_eval/results/{database}_{model_name}.json', 'w'), indent=4)

