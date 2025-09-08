import numpy as np
import torch
import pandas as pd
import scanpy as sc
import pickle
from sklearn.metrics import f1_score
from scipy.sparse import load_npz

from egoal.reasoner import RegulatoryKB
from egoal.learner_refl import ReflectLearner

data_name = 'dixit'

device = 'cuda'

Y = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()
X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()


KB = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz', neg_trn_pth=f'rules/human/{data_name}_KB_N.npz', device=device)
KB.closure_(T=5, closure_type='weighted')


Y_deduction = KB.deduce(torch.tensor(X).float().to(device)).to('cpu').numpy()

KB.closure_(T=5, closure_type='naive')
adj_matrix = torch.round(torch.abs(KB.get_KB()))
learner = ReflectLearner(input_dim= X.shape[1], output_dim= Y.shape[1], hidden_dim= 64, base_learner_type= 'GNN', adj_matrix= adj_matrix, device='cuda')
learner.load('models/human_Aug12_GNN_r_warmup.pt')

test_idx = np.load(f'dataset/human/{data_name}_test_idx.npy')
X_test = X[test_idx]
Y_test = Y[test_idx]


adj_matrix = torch.round(torch.clamp(torch.abs(KB.Regu_N_0 + KB.Regu_P_0), 0,1))
learner = ReflectLearner(input_dim= X.shape[1],
                         output_dim= Y.shape[1],
                         hidden_dim= 64,
                         base_learner_type= 'GNN',
                         adj_matrix= adj_matrix,
                         device=device)
learner.load('models/GNN_dixit_sep5_1.pt')


#test_df = pd.read_csv(f'dataset/human/{data_name}_test_set.csv', index_col=0)
#series = test_df[test_df['test_type']=='seen_2_pert'].apply(lambda x: list(range(x['data_start_idx'],x['data_end_idx+1'])), axis=1)
#pert_idx = sum(series, [])
#X_test = X_test[pert_idx]
#Y_test = Y_test[pert_idx]

Y_p = learner.predict(torch.tensor(X_test).float().to(device)).to('cpu').numpy()
R = learner.reflection(torch.tensor(X_test).float().to(device)).to('cpu').numpy().astype(bool)
#Y_d = np.load('data_anal/abduction_results/Yd_ABL0_hsa.npy')
#R = np.load('data_anal/abduction_results/R_ABL0_hsa.npy')
Y_d = Y_deduction[test_idx]

print(Y_p.shape)

#Y_p = Y_p[pert_idx]
#Y_deduction = Y_deduction[pert_idx]
#R = R[pert_idx]

total = len(Y)

#gt_con_idx = (np.nonzero(np.sum((Y_d!= Y_true) | (Y_p != Y_true), axis=0) / total < .2)[0].tolist())
print('consistent 1:',np.sum((Y_deduction==1)&(Y==Y_deduction)))
print('consistent -1:',np.sum((Y_deduction==-1)&(Y==Y_deduction)))

#kb_con_idx = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y), axis=0) / total > .3)[0].tolist()) # norman
#kb_con_idx_0 = (np.nonzero(np.sum((Y_deduction == 0) & (Y_deduction == Y), axis=0) / total > .3)[0].tolist()) # norman
#kb_con_idx = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y), axis=0) / total > .23)[0].tolist()) # adamson
#kb_con_idx_0 = (np.nonzero(np.sum((Y_deduction == 0) & (Y_deduction == Y), axis=0) / total > .2)[0].tolist()) # adamson
kb_con_idx = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y), axis=0) / total > .21)[0].tolist()) # dixit
kb_con_idx_0 = (np.nonzero(np.sum((Y_deduction == 0) & (Y_deduction == Y), axis=0) / total > .3)[0].tolist()) # dixit
kb_con_idx_1 = (np.nonzero((np.sum((Y_deduction == 1), axis=0) / total > .5) | (np.sum((Y_deduction == -1), axis=0) / total > .5))[0].tolist())

data_idx = np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y), axis=1) > 150)[0].tolist()
print(len(data_idx))

#with open('tmp_dict.txt', 'r') as f:
#    pert_f1 = eval(f.read())

#print(sorted(pert_f1.values())[-40:])
#test_perts = [k for k,v in pert_f1.items() if v > .32]

metadata = pd.read_csv(f'dataset/human/{data_name}_metadata.csv',index_col=0)
#cons_pert_idx = metadata.apply(lambda x: np.sum((np.array(data_idx) >= x['data_start_idx']) & (np.array(data_idx) < x['data_end_idx+1'])) > .7 * (x['data_end_idx+1']-x['data_start_idx']), axis=1)
#metadata_test = metadata[metadata['pert'].isin(test_perts)]
#metadata_test.to_csv(f'dataset/human/{data_name}_test_set.csv')
#print(metadata_test)

#pert_f1 = {}
#for idx,row in metadata.iterrows():
#    if row['pert'] == str(['ctrl']):
#        continue
#    pert_idx = list(range(row['data_start_idx'], row['data_end_idx+1']))
#    pert_f1[str(row['pert'])] = f1_score(Y[pert_idx].flatten(), Y_deduction[pert_idx].flatten(), average='macro')
#
#with open('tmp_dict.txt', 'w') as f:
#    f.write(str(pert_f1))

#Y_test, Y_p, Y_d, R = Y[data_idx], Y_p[data_idx], Y_d[data_idx], R[data_idx]
#Y_p, Y_d, R =  Y_p[test_idx], Y_d[test_idx], R[test_idx]

#print(f'Consistent labels with Y_true\nsra: {len(labels_gt_con)}\n')
print(f'Consistent labels with KB: {len(kb_con_idx)}\n')


' mask index for labels consistent with kb '
mask_kb = np.zeros_like(Y_test, dtype=bool)
mask_kb[:,kb_con_idx] = True
Y_mask_kb = np.where(mask_kb, Y_d, Y_p)

Y_r = np.where(R, Y_d, Y_p)

print(f'f1 of Y_deduction: {f1_score(Y_test.flatten(), Y_d.flatten(), average="macro")}')

size_y = Y.shape[0]*Y.shape[1]
size_data = np.count_nonzero(np.sum(Y, axis=1))
size_klg = .5*(np.count_nonzero(np.sum(KB.KB.cpu().numpy(), axis=1)) + np.count_nonzero(np.sum(KB.KB.cpu().numpy(), axis=1))) 
print(size_data, size_klg)
q_data = (np.count_nonzero(Y) / size_y) +\
        (size_data/(size_klg+size_data)) +\
        (np.count_nonzero((np.sum(Y, axis=1)!=0) & (np.sum(Y_deduction,axis=1)==0)) / len(Y))
q_knowledge = (np.count_nonzero(Y_deduction) / size_y) +\
        (size_klg/(size_klg+size_data)) +\
        (np.count_nonzero((np.sum(Y, axis=1)==0) & (np.sum(Y_deduction,axis=1)!=0)) / len(Y))
w_data = q_knowledge / (q_data + q_knowledge)
w_knowledge = q_data / (q_data + q_knowledge)

print(f'eval weight: data {w_data: .4f}, kb {w_knowledge: .4f}')

f1_test = f1_score(Y_test.flatten(), Y_p.flatten(), average='macro')
f1_deduc = f1_score(Y_d.flatten(), Y_p.flatten(), average='macro')
print(f'f1 of Y_p on Y_t: {f1_test: .4f}, on Y_d: {f1_deduc: .4f}, weighted: {w_data*f1_test + w_knowledge*f1_deduc: .4f}')

f1_test = f1_score(Y_test.flatten(), Y_mask_kb.flatten(), average='macro')
f1_deduc = f1_score(Y_d.flatten(), Y_mask_kb.flatten(), average='macro')
print(f'f1 of Y_m on Y_t: {f1_test: .4f}, on Y_d: {f1_deduc: .4f}, weighted: {w_data*f1_test + w_knowledge*f1_deduc: .4f}')

f1_test = f1_score(Y_test.flatten(), Y_r.flatten(), average='macro')
f1_deduc = f1_score(Y_d.flatten(), Y_r.flatten(), average='macro')
print(f'f1 of Y_r on Y_t: {f1_test: .4f}, on Y_d: {f1_deduc: .4f}, weighted: {w_data*f1_test + w_knowledge*f1_deduc: .4f}')



' weight with GO annotation '
gene2go = pickle.load(open('../GEARS/gene2go_all.pkl', 'rb'))
df_go= pd.read_csv(f'../GEARS/{data_name}/go.csv')
ann_data = sc.read_h5ad(f'../GEARS/{data_name}/perturb_processed.h5ad')
df_genes= pd.DataFrame(ann_data.var)

go_annot_num = np.array([len(gene2go[g]) if g in gene2go else 0 for g in df_genes['gene_name']], dtype=np.float32)
go_annot_num = go_annot_num / np.max(go_annot_num)
#go_annot_num = np.max(go_annot_num - .3, np.zeros_like(go_annot_num))
#print(go_annot_num)
y_mask_go = np.where((go_annot_num > .01) & (go_annot_num <= 1.), Y_d, Y_p)
print('f1 of Y_go_weight:', f1_score(Y_test.flatten(), y_mask_go.flatten(), average='macro'))


' weight with in-degree in GRN '
regulatory = load_npz(f'rules/human/{data_name}_KB_P.npz').toarray()
regulatory_num = np.sum(regulatory, axis=0)
regulatory_num = regulatory_num / np.max(regulatory_num)
#regulatory_num = np.max(regulatory_num - .3, np.zeros_like(regulatory_num))
#print(regulatory_num)
y_mask_regu = np.where((regulatory_num > .01) & (regulatory_num <= 1.), Y_d, Y_p)
print('f1 of Y_grn_weight:', f1_score(Y_test.flatten(), y_mask_regu.flatten(), average='macro'))


' get label weight '
weights = np.full(shape=Y_test.shape[1], fill_value=.1, dtype=np.float32)
#weights += (go_annot_num - .2) + (regulatory_num - .1) # norman
#weights += (go_annot_num - .3) + (regulatory_num - .2) # adamson
weights += (go_annot_num - .35) + (regulatory_num - .25) # dixit
weights[kb_con_idx] += .8
weights[kb_con_idx_0] += .3
weights[kb_con_idx_1] += .2
weights = np.clip(weights, 0., 1.)
#print(weights)
print('w >= .5:', np.count_nonzero(weights >= .5))
np.save(f'dataset/human/{data_name}_label_weight.npy', weights)


Y_w = np.where(weights>=.5, Y_d, Y_p)
f1_test = f1_score(Y_test.flatten(), Y_w.flatten(), average='macro')
f1_deduc = f1_score(Y_d.flatten(), Y_w.flatten(), average='macro')
print(f'f1 of Y_w on Y_t: {f1_test: .4f}, on Y_d: {f1_deduc: .4f}, weighted: {w_data*f1_test + w_knowledge*f1_deduc: .4f}')
