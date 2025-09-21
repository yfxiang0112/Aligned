import networkx as nx
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, confusion_matrix
from collections import defaultdict
import copy
import torch
import json
import pandas as pd
import gseapy
from tqdm import tqdm

from egoal.reasoner import RegulatoryKB

def network_diffusion_scores(G, seed_genes, alpha=0.85, tol=1e-7, max_iter=1000):
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


def gene_set_recovery(G, gene_sets, all_genes=None,  alpha=0.85):
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
        y_pred = []
        for node in all_genes:
            y_true.append(1 if node in genes_in_net else 0)
            y_score.append(true_scores.get(node, 0.0))
            y_pred.append(int(true_scores.get(node, 0.0) >= .01))
        y_true, y_score, y_pred = np.array(y_true), np.array(y_score), np.array(y_pred)
        w = np.sum(y_true == 0) / len(y_true)

        # Compute metrics on the true scores
        auroc_true = roc_auc_score(y_true, y_score)
        auprc_weighted = average_precision_score(y_true, y_score, sample_weight=np.where(y_true==1, w, 1-w))
        auprc_pos = average_precision_score(y_true, y_score, sample_weight=np.where(y_true==1, 1.,0.))
        #f1 = f1_score(y_true, y_pred)
        #confusion = confusion_matrix(y_true, y_pred)

        results[setname] = {
            'AUROC_true': auroc_true,
            'AUPRC_weighted': auprc_weighted,
            'AUPRC_pos': auprc_pos,
            #'f1': f1,
            #'confusion': str(confusion),
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

if __name__ == "__main__":
    p_lst = [0.0, .05, .1, .2, .3, .4, .5, .7, .9]
    repl_num = 3
    score_types = ['mean_auroc', 'stde_auroc', 'mean_auprc_w', 'stde_auprc_w', 'mean_auprc_p', 'stde_auprc_p']

    ann = pd.read_csv(f'dataset/human/norman_gene_ann.csv')
    genes = list(ann['gene_name'])
    gene_sets = get_gene_sets(genes,
                              database_lst = ['KEGG_2021_Human'])

    KB = RegulatoryKB(pos_trn_pth=f'scripts/klg_refine/omnipath_P.npz',
                  neg_trn_pth=f'scripts/klg_refine/omnipath_N.npz',
                  device='cpu')


    adj = torch.clamp(torch.abs(KB.Regu_P_0) + torch.abs(KB.Regu_N_0), 0,1).numpy()
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, dict(enumerate(genes)))
    res = gene_set_recovery(G, gene_sets, all_genes=genes, alpha=0.85)
    pathways = res.keys()

    results = {k: [v['AUROC_true'], 0.,
                   v['AUPRC_weighted'], 0.,
                   v['AUPRC_pos'],      0.] for k,v in res.items()}
    results['p_incomp'] = sum([[x]*len(score_types) for x in ['orig']+p_lst], [])
    results['score_type'] = score_types * (len(p_lst)+1)
    print(len(results['p_incomp']), len(results['score_type']))


    for p_incomp in p_lst:
        auroc, auprc_w, auprc_p = {k:[] for k in pathways},\
                {k:[] for k in pathways},{k:[] for k in pathways}

        for repl in range(repl_num):
            print(f'--- Processing p = {p_incomp}, replicate {repl+1} ---')

            load_model_pth =\
                    f'data_anal/refine/models/restored_{p_incomp}_mix_{repl+1}.npz'
            KB.load(load_model_pth)

            adj = torch.clamp(torch.abs(KB.Regu_P_0) + torch.abs(KB.Regu_N_0), 0,1).numpy()
            G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
            G = nx.relabel_nodes(G, dict(enumerate(genes)))


            res = gene_set_recovery(G, gene_sets, all_genes=genes, alpha=0.85)
            for k in pathways:
                auroc[k].append(res[k]['AUROC_true'])
                auprc_w[k].append(res[k]['AUPRC_weighted'])
                auprc_p[k].append(res[k]['AUPRC_pos'])

        for k in pathways:
            results[k] += [.5*(max(auroc[k])+min(auroc[k])),
                           .5*(max(auroc[k])-min(auroc[k])),

                           .5*(max(auprc_w[k])+min(auprc_w[k])),
                           .5*(max(auprc_w[k])-min(auprc_w[k])),

                           .5*(max(auprc_p[k])+min(auprc_p[k])),
                           .5*(max(auprc_p[k])-min(auprc_p[k]))]

    results = pd.DataFrame(results).set_index(['p_incomp','score_type'])
    print(results)
    results.to_csv('data_anal/refine/gsr_mix.csv')

