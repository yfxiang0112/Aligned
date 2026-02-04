import pandas as pd
import numpy as np
from scipy.sparse import load_npz

gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
print(len([i for i in gene_idx['precise1k_idx'] if i != -1]))
idx_lst = [i for i,v in enumerate(gene_idx['precise1k_idx']) if v != -1]


refined = np.round(load_npz('scripts/klg_refine/X_opt.npz').toarray())
print(refined.shape)
refined_num = np.sum(refined, axis=1)

regu_pos = load_npz('rules/regu_pos.npz').toarray()[idx_lst][:,idx_lst]
regu_neg = load_npz('rules/regu_neg.npz').toarray()[idx_lst][:,idx_lst]
print(regu_pos.shape)
regulatory_num = np.sum(np.clip(regu_pos+regu_neg, 0,1), axis=1)

closure_pos = load_npz('rules/regu_neg_clo.npz').toarray()[idx_lst][:,idx_lst]
closure_neg = load_npz('rules/regu_neg_clo.npz').toarray()[idx_lst][:,idx_lst]
print(closure_pos.shape)
closure_num = np.sum(np.clip(closure_pos+closure_neg, 0,1), axis=1)
print(closure_num)

modification = np.abs(np.clip(regu_pos+regu_neg, 0,1) - refined)
modification_num = np.sum(modification, axis=1)

regulators = {'locus': gene_idx.loc[idx_lst, 'locus'], 'symbol': gene_idx.loc[idx_lst, 'symbol'], 'regulatory': regulatory_num, 'closure':closure_num, 'refined':refined_num, 'modification':modification_num}
regulators = pd.DataFrame(regulators)

precise1k_regulators = pd.read_csv('scripts/misc/list_expe_genes/pre1k_regulators.csv')
regulators['isin_precise1k'] = regulators['locus'].isin(set(precise1k_regulators['locus']))
#print(regulators[(regulators['closure'] < regulators['refined']) & (regulators['refined']>30)])
#print(regulators[(regulators['refined'] - regulators['regulatory'] >= 30)])
print(regulators[regulators['modification'] >= 50])

regulators.loc[regulators['modification'] >= 50, ['locus','symbol','isin_precise1k']].to_csv('scripts/misc/list_expe_genes/perturbations_refined.csv')
#print(regulators[regulators['closure'] >= 50])

