import torch
import numpy as np
from egoal.reasoner import RegulatoryKB
from egoal.learner_refl import ReflectLearner
from scipy.sparse import load_npz
import pandas as pd
import json
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import json

device = 'cuda:0'
res_all = {}
for data_name in ['norman']:#, 'adamson', 'dixit']:
    #TODO
    KB_corum =   torch.tensor(load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()).float().to(device)
    KB_string =  torch.tensor(load_npz('data_anal/xref_csbench/networks/string.npz').toarray()).float().to(device)
    KB_chipseq = torch.tensor(load_npz('data_anal/xref_csbench/networks/chipseq.npz').toarray()).float().to(device)
    KB_lr =      torch.tensor(load_npz('data_anal/xref_csbench/networks/lr_pairs.npz').toarray()).float().to(device)

    for model_type in ['GNN', 'MLP']:

        X = torch.tensor(load_npz(f'dataset/human/{data_name}_X.npz').toarray(), dtype=torch.float32)
        Y = torch.tensor(load_npz(f'dataset/human/{data_name}_Y.npz').toarray(), dtype=int)
        total = len(X)
        
        reasoner = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                          neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
                          device=device)
        reasoner.closure_(T=5, closure_type='weighted')
        
        adj_matrix = torch.round(torch.clamp(torch.abs(reasoner.Regu_N_0 + reasoner.Regu_P_0), 0,1))
        learner = ReflectLearner(input_dim= X.shape[1],
                                 output_dim= Y.shape[1],
                                 hidden_dim= 64,
                                 base_learner_type= model_type,
                                 adj_matrix= adj_matrix,
                                 device=device)
        
        label_weight = torch.tensor(np.load(f'dataset/human/{data_name}_label_weight.npy')).to(device)
        print('w>.5: ', torch.count_nonzero(label_weight>=.5).item())
        
        res = {'baseline':{'corum':[], 'corum_confusion':[], 'string':[], 'string_confusion':[], 'chipseq':[], 'chipseq_confusion':[], 'lr':[], 'lr_confusion':[]},
               'align1':  {'corum':[], 'corum_confusion':[], 'string':[], 'string_confusion':[], 'chipseq':[], 'chipseq_confusion':[], 'lr':[], 'lr_confusion':[]},
               'refine1': {'corum':[], 'corum_confusion':[], 'string':[], 'string_confusion':[], 'chipseq':[], 'chipseq_confusion':[], 'lr':[], 'lr_confusion':[]},}
        
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

                Y_d_corum   = torch.clamp((X_test @ KB_corum  ), -1.,1.).int().cpu().numpy().flatten()
                Y_d_string  = torch.clamp((X_test @ KB_string ), -1.,1.).int().cpu().numpy().flatten()
                Y_d_chipseq = torch.clamp((X_test @ KB_chipseq), -1.,1.).int().cpu().numpy().flatten()
                Y_d_lr      = torch.clamp((X_test @ KB_lr     ), -1.,1.).int().cpu().numpy().flatten()
        
                cf_corum   = confusion_matrix(np.abs(Y_d_corum),   np.abs(Y_r), labels=[1,0])
                cf_string  = confusion_matrix(np.abs(Y_d_string),  np.abs(Y_r), labels=[1,0])
                cf_chipseq = confusion_matrix(np.abs(Y_d_chipseq), np.abs(Y_r), labels=[1,0])
                cf_lr      = confusion_matrix(np.abs(Y_d_lr),      np.abs(Y_r), labels=[1,0])

                f1_corum   = f1_score(np.abs(Y_d_corum)  , np.abs(Y_r))
                f1_string  = f1_score(np.abs(Y_d_string) , np.abs(Y_r))
                f1_chipseq = f1_score(np.abs(Y_d_chipseq), np.abs(Y_r))
                f1_lr      = f1_score(np.abs(Y_d_lr)     , np.abs(Y_r))

                res[stage]['corum_confusion'  ].append(cf_corum.tolist())
                res[stage]['string_confusion' ].append(cf_string.tolist())
                res[stage]['chipseq_confusion'].append(cf_chipseq.tolist())
                res[stage]['lr_confusion'     ].append(cf_lr.tolist())
                res[stage]['corum'  ].append(f1_corum)
                res[stage]['string' ].append(f1_string)
                res[stage]['chipseq'].append(f1_chipseq)
                res[stage]['lr'     ].append(f1_string)
        
        
                del X_test, Y_test, Y_p, Y_r, R, Y_d
                torch.cuda.empty_cache()
        

        res_mean = {k: {k1: .5*(max(v1)+min(v1)) for k1,v1 in v.items() if 'confusion' not in k1} for k,v in res.items()}
        res_stde = {k: {k1: .5*(max(v1)-min(v1)) for k1,v1 in v.items() if 'confusion' not in k1} for k,v in res.items()}
        res_final = {'mean':res_mean, 'stde':res_stde, 'orig':res}
        res_all[f'{data_name}_{model_type}'] = res_final
    
json.dump(res_all, open(f'data_anal/model_als/xref_cons.json', 'w'), indent=4)
