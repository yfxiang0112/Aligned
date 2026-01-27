from cobra.core import gene
from egoal.reasoner import RegualtoryKB
import pandas as pd

gene_idx = pd.read_csv('dataset/gene_idx.csv',index_col=0)

regulatoryKB = RegualtoryKB(pos_trn_pth='rules/pos_regu', neg_trn_pth='rules/neg_regu')
d = {}
for i,j in zip(regulatoryKB.T_P.to_lists()[0], regulatoryKB.T_P.to_lists()[1]):
    if i in d:
        d[i].add(j)
    else:
        d[i] = set([j])
for i,j in zip(regulatoryKB.T_N.to_lists()[0], regulatoryKB.T_N.to_lists()[1]):
    if i in d:
        d[i].add(j)
    else:
        d[i] = set([j])
idx = [i for i,s in d.items() if len(s)>50]

data_regulators = ["arcZ","cyaR","gcvB","micA","ryhB","rydC"]
for i,s in d.items():
    if gene_idx.loc[i,'symbol'] in data_regulators:
        print(gene_idx.loc[i,'symbol'],len(s))


regulators = gene_idx.iloc[idx,:].loc[:,['symbol','locus']]
print(list(regulators['symbol']))
regulators.to_csv('rules/regulators_ecocyc.csv', index=False)
