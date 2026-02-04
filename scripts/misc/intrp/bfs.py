import torch
import numpy as np
import pandas as pd
from aligned.reasoner import RegulatoryKB
from aligned.learner_adap import AdaptorLearner
from scipy.sparse import load_npz
import json
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from collections import deque

def bfs_subtree_edges(adj, root, max_depth):
	'''
	adj: torch.Tensor (N x N), edge labels {0,1,2,3}
	root: int, starting node index
	max_depth: int, BFS depth limit
	
	returns: pandas.DataFrame with columns
	[source, target, edge_label, depth]
	'''

	assert adj.dim() == 2 and adj.size(0) == adj.size(1)
	N = adj.size(0)

	visited = set([root])
	queue = deque([(root, 0)])
	edges = []

	while queue:
		u, depth = queue.popleft()
		if depth == max_depth:
			continue

		# find outgoing edges u -> v
		out_edges = torch.nonzero(adj[u] != 0, as_tuple=False)

		for v_tensor in out_edges:
			v = int(v_tensor.item())
			label = int(adj[u, v].item())

			edges.append({
				"src": u,
				"tar": v,
				"sign": '+' if label==1 else '-' if (label==2 or label==-1) else '+-',
				"depth": depth + 1
			})

			if v not in visited:
				visited.add(v)
				queue.append((v, depth + 1))

	return pd.DataFrame(edges)


data_name = 'norman'
device = 'cuda'
k = 5
gene_idx = pd.read_csv('dataset/human/norman_gene_ann.csv',index_col=0)
gene_ann = pd.read_csv('dataset/human/norman_gene_ann.csv',index_col=1)

############################################
# --- 1. prepare original & refined KB ---
reasoner = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_omnipath_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_omnipath_KB_N.npz',
                  device=device)
reasoner.closure_(T=5, closure_type='weighted')

adj_orig = reasoner.Regu_P_0 + 2* reasoner.Regu_N_0

reasoner.load(f'results/ex1_aligned/{data_name}/models/GNN_abl0_2.npz')
#adj_refn = reasoner.Regu_P_0 + 2* reasoner.Regu_N_0
adj_refn = reasoner.KB_P + 2* reasoner.KB_N

#############################################
# --- 2. prepare learner ---
d = reasoner.Regu_P_0.shape[0]
adj_matrix = torch.round(torch.clamp(
        torch.tensor(load_npz(f'rules/human/{data_name}_KB_P.npz').toarray()).to(device) +
        torch.tensor(load_npz(f'rules/human/{data_name}_KB_N.npz').toarray()).to(device) , 0,1))
learner = AdaptorLearner(input_dim= d,
                         output_dim= d,
                         hidden_dim= 64,
                         base_learner_type= 'GNN',
                         adj_matrix= adj_matrix,
                         device=device)
X = torch.eye(d).to(device).float()
learner.load(f'results/ex1_aligned/{data_name}/models/GNN_abl0_2.pt')


Y_p = learner.predict(X).float()
_,R = learner.forward(X)
#Y_d= reasoner.deduce(torch.tensor(X).float().to(device)).float()
#Y_r = torch.where(R>=.5, Y_d, Y_p).float()

#########################################
# --- 3. run bfs ---

df_lst = []
roots = ['CNN1', 'CBL']
for r in roots:
    root_idx = gene_idx['vector_idx'][r]
    tree_orig = bfs_subtree_edges(adj_orig, root_idx, k)
    tree_orig['src_gene'] = tree_orig['src'].apply(lambda x: gene_ann.loc[x])
    tree_orig['tar_gene'] = tree_orig['tar'].apply(lambda x: gene_ann.loc[x])
    tree_orig['root'] = r
    tree_orig['conf'] = tree_orig.apply(lambda row: R[row['src'],row['tar']].item(), axis=1)
    tree_orig['pred'] = tree_orig.apply(lambda row: Y_p[row['src'],row['tar']].item(), axis=1)

    tree_refn = bfs_subtree_edges(adj_refn, root_idx, k)
    tree_refn['src_gene'] = tree_refn['src'].apply(lambda x: gene_ann.loc[x])
    tree_refn['tar_gene'] = tree_refn['tar'].apply(lambda x: gene_ann.loc[x])
    tree_refn['root'] = r
    tree_refn['conf'] = tree_refn.apply(lambda row: R[row['src'],row['tar']].item(), axis=1)
    tree_refn['pred'] = tree_refn.apply(lambda row: Y_p[row['src'],row['tar']].item(), axis=1)

    df_lst.append(tree_orig)
    df_lst.append(tree_refn)

df_lst = pd.concat(df_lst, axis=0)
df_lst.to_csv('scripts/misc/intrp/tree.csv', index=False)
