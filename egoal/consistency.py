import pygraphblas as pgb
import pandas as pd
import numpy as np
import json
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt, parameter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

from reasoner import RegualtoryKB, MetabolicKB
from utils import matrix2pgb, optvec2pgb, xor, pgb2ternary

' load predicted labels '
df_pred = pd.read_csv('data_anal/cross_val.tsv', sep='\t').sort_values('data_idx').dropna(axis=0)
df_pred['data_idx'] = df_pred['data_idx'].astype(int)
df_pred = df_pred[df_pred['type']=='pred'].set_index('data_idx').loc[:,'b0002':]

T_P = pgb.Matrix.from_binfile('rules/pos_gem')
T_N = pgb.Matrix.from_binfile('rules/neg_gem')
P2_idx = set((T_P@T_P@T_P)[:,0].to_lists()[0])
N_idx = set((T_N)[:,0].to_lists()[0])
print(len(P2_idx-N_idx), len(N_idx-P2_idx))

TRN_KB = RegualtoryKB('rules/pos_regu', 'rules/neg_regu')
GEM_KB = MetabolicKB('rules/pos_gem', 'rules/neg_gem', 'rules/gem_annot', T=3)

M_P = GEM_KB.A.cast(pgb.BOOL) @ GEM_KB.T_P.cast(pgb.BOOL)
M_N = GEM_KB.A.cast(pgb.BOOL) @ GEM_KB.T_N.cast(pgb.BOOL)
#print(M_P.shape)
biomass_P = M_P.extract_matrix(range(M_P.shape[0]-1), [0]).cast(pgb.BOOL)
biomass_N = M_N.extract_matrix(range(M_N.shape[0]-1), [0]).cast(pgb.BOOL)
#print(biomass_P.to_lists()[1], biomass_N.to_lists()[1])
pos = biomass_P.to_lists()[0]
neg = biomass_N.to_lists()[0]
print(len(pos), len(neg))
print(f'len P-N: {len(set(pos) - set(neg))}, N-P: {len(set(neg)-set(pos))}')
#print(sorted(list(pos-neg)))
#print(sorted(list(neg-pos)))
#print(sorted(list(neg)))

df_trn = pd.read_csv('data_anal/incons_trn.tsv', sep='\t')
df_meta = pd.read_csv('dataset/metadata_gene.csv',index_col=0)

test = df_trn.loc[df_trn['type']=='test',:].iloc[:,4:].reset_index(drop=True)
trn_p = df_trn.loc[df_trn['type']=='kb+',:].iloc[:,4:].reset_index(drop=True)
trn_n = df_trn.loc[df_trn['type']=='kb-',:].iloc[:,4:].reset_index(drop=True)

print(((test==trn_p) & (trn_n==0)) | ((test==trn_p) & (trn_n==0)))
index_trn = test.columns[(((test==trn_p) & (trn_n==0)) | ((test==trn_n) & (trn_p==0))).sum(axis=0)>30]

#print((trn_p+trn_n == 0).sum(axis=0) > 50)
test_p = test.iloc[:,pos]#.loc[:, (trn_p+trn_n == 0).sum(axis=0) > 50]
test_n = test.iloc[:,neg]#.loc[:, (trn_p+trn_n == 0).sum(axis=0) > 50]

regulators = ["arcZ","cyaR","gcvB","micA","ryhB","rydC"]
#regulators = ["cyaR","gcvB","micA","ryhB","rydC"]
p_growth = sorted(sum([eval(lst) for lst in\
        df_meta.loc[(df_meta['fitness']>1)&(df_meta.index.isin(regulators)),'dataset_items']],[]))
n_growth = sorted(sum([eval(lst) for lst in\
        df_meta.loc[(df_meta['fitness']<1)&(df_meta.index.isin(regulators)),'dataset_items']],[]))

print(n_growth)
index_p = test_p.columns[((test_p.iloc[n_growth]<0).sum(axis=0)\
                                    + (test_p.iloc[p_growth]>0).sum(axis=0))\
                                    / (len(p_growth)+len(n_growth)) > .2]
#index_p_2 = test_p.columns[((test_p.iloc[p_growth]>0).sum(axis=0)\
#                                    + (test_p.iloc[n_growth]<=0).sum(axis=0))\
#                                    / (len(p_growth)+len(n_growth)) > .2]
#print(len(index_p_1), len(index_p_2))

index_n = test_n.columns[((test_n.iloc[n_growth]>0).sum(axis=0)\
                                    + (test_n.iloc[p_growth]<0).sum(axis=0))\
                                    / (len(p_growth)+len(n_growth)) > .3]
#index = [i for i,v in enumerate(test.columns) if v in index_p or v in index_n or v in index_trn]
#print(len(index))
#print(index)

print(len(index_p), len(index_n))
print([i for i,v in enumerate(test.columns) if v in index_p])
#print([i for i,v in enumerate(test.columns) if v in index_p or v in index_n])

exit()
#print([i for i,v in enumerate(test.columns) if v in index_p])
#print([i for i,v in enumerate(test.columns) if v in index_n])
#print([i for i,v in enumerate(test.columns) if v in index_p or v in index_n])

# NOTE tmp
#label_set = pd.read_csv('dataset/label_set_1096.csv', index_col=0)
#label_set = label_set.iloc[index].reset_index(drop=True)
#label_set.to_csv('dataset/label_set.csv')

#print(((test.iloc[p_growth]==1)|(test.iloc[n_growth]==-1)).sum(axis=0).mean())
#print(((test.iloc[n_growth]==1)|(test.iloc[p_growth]==-1)).sum(axis=0).mean())






#def fba_result(z,n_cols):
#    max_modify = 100
#    budget = 2000
#    Y = pgb.Vector.sparse(size=n_cols, typ=pgb.BOOL)
#    opt_dim = Dimension(size= max_modify * 2,
#                        regs= [[-1, n_cols-1]] * (max_modify * 2),
#                        tys= [False] * (max_modify * 2))
#
#    ''' def opt obj & soluttion to pgb matrix '''
#    def objective(I):
#        I_pos, I_neg = optvec2pgb(I, 1, n_cols,
#                                  return_matrix=False, max_modify= max_modify)
#        y_pos_ = xor(I_pos, Y)
#        y_neg_ = xor(I_neg, Y)
#        return fluxKB.violated(y_pos_, y_neg_, z)
#    opt_obj = Objective(objective, opt_dim)
#    solution = Opt.min(opt_obj, Parameter(budget=budget, seed=42))
#    I_pos, I_neg = optvec2pgb(solution, 1, n_cols, return_matrix=False, max_modify= max_modify)
#    return I_pos, I_neg


''' test for labeled data '''
X_label = np.load('dataset/X_label.npy')
Y_label = np.load('dataset/Y_train.npy')
semisup_df = pd.read_csv('dataset/X_semisup.csv', index_col=0)['growth']


gene_mapping = (pd.read_csv('dataset/raw/genome_annotations.tsv', sep='\t')['Locus tag']).to_dict()
gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
locus2symbol = {locus:symbol for locus,symbol in zip(gene_idx['locus'],gene_idx['symbol'])}
label_set = pd.read_csv('dataset/label_set.csv', index_col=0)['locus']
#label_set = pd.read_csv('dataset/counts.csv').columns[1:]



' write headers '
#with open('data_anal/incons_growth.tsv', 'w') as f:
#    f.write('gene_idx\tlocus\ttest_growth\tpred_growth\tground_growth\n')

with open('data_anal/incons_trn.tsv', 'w') as f:
    f.write('data_idx\ttype\tgene_idx\tlocus')
    for locus in label_set:
        f.write(f'\t{locus}')
    f.write('\n')

#with open('data_anal/incons_fba.tsv', 'w') as f:
#    f.write('data_idx\ttype\tgene_idx\tlocus')
#    for locus in label_set:
#        f.write(f'\t{locus}')
#    f.write('\n')


' iterate on data '
for data_idx, x,y in tqdm(zip(range(len(X_label)), X_label, Y_label),total=70):
    y_pos = matrix2pgb(y== 1)
    y_neg = matrix2pgb(y==-1)
    overexpr_idx = np.nonzero(x)[0]
    x = matrix2pgb(x)

    if len(overexpr_idx) == 0:
        gt_growth = 1.
        overexpr_idx = -1
    elif len(overexpr_idx) == 1:
        if overexpr_idx[0] not in semisup_df.index:
            gt_growth = 1.
        else:
            gt_growth = semisup_df[overexpr_idx[0]]
        overexpr_idx = overexpr_idx[0]
    else:
        continue

    #if overexpr_idx != -1:
    #    y_p_pos = matrix2pgb(df_pred.loc[data_idx].to_numpy() == 1)
    #    y_p_neg = matrix2pgb(df_pred.loc[data_idx].to_numpy() == -1)
    #    pred_growth = fluxKB.pseudo_ratio(y_p_pos,y_p_neg)
    #else:
    #    pred_growth = 1.
    pred_growth = 1.



    ' gt growth rate '
    #with open('data_anal/incons_growth.tsv', 'a') as f:
    #    f.write(f"{overexpr_idx}\t{'WT' if overexpr_idx==-1 else gene_mapping[overexpr_idx]}\t{fluxKB.pseudo_ratio(y_pos, y_neg)}\t{pred_growth}\t{gt_growth}\n")


    ' TRN deduction result '
    y_pos_, y_neg_ = np.zeros_like(y), np.zeros_like(y)
    y_pos_deduce,y_neg_deduce = TRN_KB.deduce(x)
    for idx in y_pos_deduce.to_lists()[0]:
        y_pos_[idx] = 1
    for idx in y_neg_deduce.to_lists()[0]:
        y_neg_[idx] = -1

    with open('data_anal/incons_trn.tsv', 'a') as f:
        f.write(f"{data_idx}\ttest\t{overexpr_idx}\t{'WT' if overexpr_idx==-1 else gene_mapping[overexpr_idx]}")
        for y_i in y:
            f.write(f"\t{y_i}")
        f.write(f"\n{data_idx}\tkb+\t{overexpr_idx}\t{'WT' if overexpr_idx==-1 else gene_mapping[overexpr_idx]}")
        for y_i in y_pos_:
            f.write(f"\t{y_i}")
        f.write(f"\n{data_idx}\tkb-\t{overexpr_idx}\t{'WT' if overexpr_idx==-1 else gene_mapping[overexpr_idx]}")
        for y_i in y_neg_:
            f.write(f"\t{y_i}")
        f.write("\n")
    continue

    #####################################

    ' FBA zoopt result '
    y_pos_, y_neg_ = np.zeros_like(y), np.zeros_like(y)
    #print(overexpr_idx, gt_growth)
    with open('dataset/fitness.json', 'r') as f:
        fitness = json.load(f)
    gt_growth = fitness[locus2symbol[gene_mapping[overexpr_idx]]] if overexpr_idx!=-1 else 1.
    if gt_growth != 1.:
        print(gt_growth)
        y_pos_res,y_neg_res = fba_result(gt_growth, n_cols=Y_label.shape[1])
        for idx in y_pos_res.to_lists()[0]:
            y_pos_[idx] = 1
        for idx in y_neg_res.to_lists()[0]:
            y_neg_[idx] = -1
    print(y_pos_)

    with open('data_anal/incons_fba.tsv', 'a') as f:
        f.write(f"{data_idx}\ttest\t{overexpr_idx}\t{'WT' if overexpr_idx==-1 else gene_mapping[overexpr_idx]}")
        for y_i in y:
            f.write(f"\t{y_i}")
        f.write(f"\n{data_idx}\tkb+\t{overexpr_idx}\t{'WT' if overexpr_idx==-1 else gene_mapping[overexpr_idx]}")
        for y_i in y_pos_:
            f.write(f"\t{y_i}")
        f.write(f"\n{data_idx}\tkb-\t{overexpr_idx}\t{'WT' if overexpr_idx==-1 else gene_mapping[overexpr_idx]}")
        for y_i in y_neg_:
            f.write(f"\t{y_i}")
        f.write("\n")
    

#print(TRN_KB.violated(matrix2pgb(Y_label==1),matrix2pgb(Y_label==-1),matrix2pgb(X_label)))


#########################################################



#gene_idx_mapping = pd.read_csv('dataset/gene_idx.csv', index_col=0)
#
#Regu_P = pgb.Matrix.from_binfile('rules/pos_regu')
#Regu_N = pgb.Matrix.from_binfile('rules/neg_regu')
#T_P, T_N = closure(Regu_P, Regu_N)
#
#X = np.load('dataset/X_label.npy')
#Y = np.load('dataset/Y_label.npy')
#
#label_select = {}
#
#Y_trn = pgb2ternary(matrix2pgb(X)@T_P, matrix2pgb(X)@T_N).numpy()
#Y_ = Y[np.any(Y_trn!=0, axis=1)]
#Y_trn = Y_trn[np.any(Y_trn!=0, axis=1)]
#for d in range(Y.shape[1]):
#    if np.count_nonzero(Y[:,d])==0 and np.count_nonzero(Y_trn[:,d])==0:
#        continue
#    f1_macro = f1_score(Y_[:,d], Y_trn[:,d], average='macro')
#    f1_micro = f1_score(Y_[:,d], Y_trn[:,d], average='micro')
#    f1_class = f1_score(Y_[:,d], Y_trn[:,d], average=None)
#    if min(f1_class)>.2 and f1_macro>.4:
#        label_select[d] = (f1_macro, f1_micro, f1_class, Y_[:,d], Y_trn[:,d])
#
#print(len(label_select),'\n')
#locus = []
#idx = []
#for k,v in label_select.items():
#    print(f'label {k}, symbol {gene_idx_mapping.loc[k,"symbol"]}, locus {gene_idx_mapping.loc[k,"locus"]}\nmacro f1 {v[0]}, micro f1 {v[1]}, class f1 {v[2]}\ntrue {v[3]}\npred {v[4]}\n')
#    locus.append(gene_idx_mapping.loc[k,'locus'])
#    idx.append(k)
#
#label_set_df = pd.DataFrame({'locus':locus, 'matrix_idx':idx})
#label_set_df.to_csv('dataset/label_set.csv')
#np.save('dataset/Y_train.npy', Y[:,idx])
