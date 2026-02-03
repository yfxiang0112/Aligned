import torch
import numpy as np
from aligned.reasoner import RegulatoryKB
from aligned.learner_adap import AdaptorLearner
from scipy.sparse import load_npz
import pandas as pd
import json
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import json

data_name = 'adamson'
model_type = 'GNN'
device = 'cuda:0'

X = torch.tensor(load_npz(f'dataset/human/{data_name}_X.npz').toarray(), dtype=torch.float32)
Y = torch.tensor(load_npz(f'dataset/human/{data_name}_Y.npz').toarray(), dtype=int)
total = len(X)

reasoner = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                  neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                  device=device)
reasoner.closure_(T=5, closure_type='weighted')

adj_matrix = torch.round(torch.clamp(torch.abs(reasoner.Regu_N_0 + reasoner.Regu_P_0), 0,1))
learner = AdaptorLearner(input_dim= X.shape[1],
                         output_dim= Y.shape[1],
                         hidden_dim= 64,
                         base_learner_type= model_type,
                         adj_matrix= adj_matrix,
                         device=device)

label_weight = torch.tensor(np.load(f'dataset/human/{data_name}_label_weight.npy')).to(device)
print('w>.5: ', torch.count_nonzero(label_weight>=.5).item())

res = {'baseline':{'test': {'data_f1':[], 'kb_f1':[], 'data_confusion':[], 'kb_confusion':[],
                    'data_precision':[], 'data_recall':[], 'kb_precision':[], 'kb_recall':[]},
                   'train': {'data_f1':[], 'kb_f1':[], 'data_confusion':[], 'kb_confusion':[],
                    'data_precision':[], 'data_recall':[], 'kb_precision':[], 'kb_recall':[]}},
       'align1':{'test': {'data_f1':[], 'kb_f1':[], 'data_confusion':[], 'kb_confusion':[],
                    'data_precision':[], 'data_recall':[], 'kb_precision':[], 'kb_recall':[]},
                   'train': {'data_f1':[], 'kb_f1':[], 'data_confusion':[], 'kb_confusion':[],
                    'data_precision':[], 'data_recall':[], 'kb_precision':[], 'kb_recall':[]}},
       'refine1':{'test': {'data_f1':[], 'kb_f1':[], 'data_confusion':[], 'kb_confusion':[],
                    'data_precision':[], 'data_recall':[], 'kb_precision':[], 'kb_recall':[]},
                   'train': {'data_f1':[], 'kb_f1':[], 'data_confusion':[], 'kb_confusion':[],
                    'data_precision':[], 'data_recall':[], 'kb_precision':[], 'kb_recall':[]}}}

for stage, learn_model, kb_model in [('baseline','',''), ('align1','abl0_',''), ('refine1','abl0_','abl0_')]:
    for repl in range(5):

        learner.load(f'data_anal/experiment_results/{data_name}/models/{model_type}_{learn_model}{repl+1}.pt')
        if kb_model != '':
            reasoner.load(f'data_anal/experiment_results/{data_name}/models/{model_type}_{kb_model}{repl+1}.npz')

        test_idx = np.zeros(shape=len(X), dtype=bool)
        if data_name == 'norman':
            test_idx[np.load(f'dataset/human/{data_name}_test_idx.npy')] = True
        else:
            test_idx = np.load(f'data_anal/experiment_results/{data_name}/models/{model_type}_split_{repl+1}.npy')

        X_test, Y_test = X[test_idx].to(device), Y[test_idx].to(device)
        Y_p = learner.predict(torch.tensor(X_test).float().to(device)).float()
        R = learner.reflection(torch.tensor(X_test).float().to(device))
        Y_d= reasoner.deduce(torch.tensor(X_test).float().to(device)).float()
        Y_r = torch.where(R>=.5, Y_d, Y_p).float()
        Y_r, Y_test, Y_d =Y_r.to('cpu').numpy(), Y_test.to('cpu').numpy(), Y_d.to('cpu').numpy()
        Y_r, Y_test, Y_d = Y_r.flatten(), Y_test.flatten(), Y_d.flatten()

        data_confusion = confusion_matrix(Y_test, Y_r, labels=[-1,0,1])
        data_confusion = data_confusion / np.sum(data_confusion)
        res[stage]['test']['data_confusion'].append(data_confusion.tolist())
        kb_confusion = confusion_matrix(Y_d, Y_r, labels=[-1,0,1])
        kb_confusion = kb_confusion / np.sum(kb_confusion)
        res[stage]['test']['kb_confusion'].append(kb_confusion.tolist())

        data_cons = f1_score(Y_test, Y_r, average='macro')
        res[stage]['test']['data_f1'].append(data_cons)
        kb_cons = f1_score(Y_d, Y_r, average='macro')
        res[stage]['test']['kb_f1'].append(kb_cons)

        data_prec = precision_score(Y_test, Y_r, average='macro')
        data_recl = recall_score(Y_test, Y_r, average='macro')
        kb_prec = precision_score(Y_d, Y_r, average='macro')
        kb_recl = recall_score(Y_d, Y_r, average='macro')
        res[stage]['test']['data_precision'].append(data_prec)
        res[stage]['test']['data_recall'].append(data_recl)
        res[stage]['test']['kb_precision'].append(kb_prec)
        res[stage]['test']['kb_recall'].append(kb_recl)

        print(f'test set eval: data cons {data_cons}, p {data_prec}, r {data_recl},\nkb cons {kb_cons},\n confusion {data_confusion}')

        del X_test, Y_test, Y_p, Y_r, R, Y_d
        torch.cuda.empty_cache()

        X_train, Y_train = X[~test_idx].to(device), Y[~test_idx].to(device)
        
        Y_p = learner.predict(torch.tensor(X_train).float().to(device)).float()
        R = learner.reflection(torch.tensor(X_train).float().to(device))
        Y_d= reasoner.deduce(torch.tensor(X_train).float().to(device)).float()
        Y_r = torch.where(R>=.5, Y_d, Y_p).float()
        Y_r, Y_train, Y_d =Y_r.to('cpu').numpy(), Y_train.to('cpu').numpy(), Y_d.to('cpu').numpy()
        Y_r, Y_train, Y_d = Y_r.flatten(), Y_train.flatten(), Y_d.flatten()

        data_confusion = confusion_matrix(Y_train, Y_r, labels=[-1,0,1])
        data_confusion = data_confusion / np.sum(data_confusion)
        res[stage]['train']['data_confusion'].append(data_confusion.tolist())
        kb_confusion = confusion_matrix(Y_d, Y_r, labels=[-1,0,1])
        kb_confusion = kb_confusion / np.sum(kb_confusion)
        res[stage]['train']['kb_confusion'].append(kb_confusion.tolist())

        data_cons = f1_score(Y_train, Y_r, average='macro')
        res[stage]['train']['data_f1'].append(data_cons)
        kb_cons = f1_score(Y_d, Y_r, average='macro')
        res[stage]['train']['kb_f1'].append(kb_cons)

        data_prec = precision_score(Y_train, Y_r, average='macro')
        data_recl = recall_score(Y_train, Y_r, average='macro')
        kb_prec = precision_score(Y_d, Y_r, average='macro')
        kb_recl = recall_score(Y_d, Y_r, average='macro')
        res[stage]['train']['data_precision'].append(data_prec)
        res[stage]['train']['data_recall'].append(data_recl)
        res[stage]['train']['kb_precision'].append(kb_prec)
        res[stage]['train']['kb_recall'].append(kb_recl)

        print(f'train set eval: data cons {data_cons}, p {data_prec}, r {data_recl},\nkb cons {kb_cons},\n confusion {data_confusion}')

        del X_train, Y_train, Y_p, Y_r, R, Y_d
        torch.cuda.empty_cache()

res_mean = {k: {k1: {k2: .5*(max(v2)+min(v2)) for k2,v2 in v1.items() if 'confusion' not in k2} for k1,v1 in v.items()} for k,v in res.items()}
res_stde = {k: {k1: {k2: .5*(max(v2)-min(v2)) for k2,v2 in v1.items() if 'confusion' not in k2} for k1,v1 in v.items()} for k,v in res.items()}
res_final = {'mean':res_mean, 'stde':res_stde, 'orig':res}
    
json.dump(res_final, open(f'data_anal/model_als/eval_{data_name}_{model_type}.json', 'w'), indent=4)
