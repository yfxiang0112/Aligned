import pandas as pd
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import load_npz
import glob
import os

from aligned.reasoner import RegulatoryKB


def weighted_mean(f1_data, f1_kb, w):
    p_integrate = 2
    return (w * (f1_data ** -p_integrate)
            + (1.-w) * (f1_kb ** -p_integrate)) ** (-1/p_integrate)

def key_proc(x):
    if '+' not in x and x!='ctrl':
        return x
    elif x=='ctrl':
        return ''
    else:
        perts = sorted(x.split('+'))
        if 'ctrl' not in perts:
            return f'{perts[0]}_{perts[1]}'
        else:
            perts.remove('ctrl')
            return perts[0]


#data_name = 'dixit'
#model_name = 'gears'
device = 'cuda'

df_lst = []
for data_name in ['norman', 'dixit', 'adamson']:
    mse_mean_lst, data_f1_mean_lst, kb_f1_mean_lst, bal_f1_mean_lst = [], [], [], []
    mse_stde_lst, data_f1_stde_lst, kb_f1_stde_lst, bal_f1_stde_lst = [], [], [], []

    xref_kb = {'corum':   torch.tensor(load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()).float().to(device),
               'string':  torch.tensor(load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()).float().to(device),
               'chipseq': torch.tensor(load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()).float().to(device),
               'lr':      torch.tensor(load_npz('data_anal/xref_csbench/networks/corum.npz').toarray()).float().to(device)}\
              if data_name == 'norman' else {}
    xref_mean_lst, xref_stde_lst = [[],[],[],[]], [[],[],[],[]]

    ' prepare test data '
    test_metadata = pd.read_csv(
        f'dataset/human/{data_name}_test_set.csv', index_col=0)

    test_idx = np.load(f'dataset/human/{data_name}_test_idx.npy')
    Y_true = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()[test_idx]
    X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()[test_idx]
    Y_true_con = load_npz( f'dataset/human/{data_name}_Y_con.npz').toarray()[test_idx]
        
    KB = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
                      neg_trn_pth=f'rules/human/{data_name}_KB_N.npz', device=device)
    KB.closure_(T=5, closure_type='weighted')
    Y_deduction = KB.deduce(torch.tensor(
        X).float().to(device)).to('cpu').numpy()

    xref_deduction = {k: torch.clamp((torch.tensor(X).float().to(device) @ xref_KB), -1.,1.).int().cpu().numpy()\
                      for k, xref_KB in xref_kb.items()} if data_name=='norman' else {}
        
        
    model_lst =['state', 'additive', 'gears', 'scgpt', 'scfoundation'] 
    for model_name in model_lst:

        directory = f'results/ex1_baselines/{model_name}_{data_name}'
        file_pattern = 'all_predictions*.json'
        log_files = glob.glob(os.path.join(directory, file_pattern))
        log_files.sort()  # Sort for consistent ordering
        
        if not log_files:
            assert 0
        
        data_f1_lst, kb_f1_lst, bal_f1_lst = [], [], []
        mse_lst = []
        xref_f1_lst = {'corum':[], 'string':[], 'chipseq':[], 'lr':[]}
        
        for file_idx, log_file in enumerate(log_files, 1):
            print(file_idx, log_file)
            pred_result = json.load(open(log_file, 'r'))
            pred_result = {key_proc(k): v for k,v in pred_result.items()}

            #print(pred_result.keys())
        
            genome_size = len(list(pred_result.values())[0])
        
            indices = list(test_metadata.apply(lambda x: (
                x['data_start_idx'], x['data_end_idx+1']), axis=1))
            pert_keys = test_metadata['pert'].apply(
                lambda x: f'{sorted(eval(x))[0]}_{sorted(eval(x))[1]}' if 'ctrl' not in x else list(set(eval(x))-{'ctrl'})[0])

            #print(pert_keys)
            print(f'{len(set(pert_keys) - pred_result.keys())} / {len(set(pert_keys))} missing perts: {set(pert_keys) - pred_result.keys()}')
            predictions = list(pert_keys.apply(
                lambda x: pred_result[x] if x in pred_result else [0]*genome_size))
        
            # Find the maximum row index needed
            max_row = max(end for _, end in indices) if indices else 0
        
            # Find the maximum column length (assuming all predictions[i] have same length)
            max_col = max(len(a) for a in predictions) if predictions else 0
        
            # Initialize the matrix with zeros
            Y = np.zeros((max_row, max_col))
        
            # Fill the matrix according to the specifications
            for i, (start, end) in enumerate(indices):
                if i < len(predictions):
                    # Convert predictions[i] to numpy array and ensure proper shape
                    a_array = np.array(predictions[i])
                    # Repeat the array for the specified row range
                    for row in range(start, end):
                        Y[row, :len(a_array)] = a_array
            #print(Y)
        
            ' eval MSE '
            criterion = torch.nn.MSELoss(reduction='mean')
            mse_score = criterion(torch.tensor(Y), torch.tensor(Y_true_con))
            #print(f'MSE: {mse_score}')
            mse_lst.append(float(mse_score))
        
            Y = np.where(np.abs(Y) > .27, np.sign(Y), 0)
        
            ''' eval data consistency '''
            print(Y.shape)
            f1_data = f1_score(Y_true.flatten(), Y.flatten(), average="macro")
            #print(f'data f1: {f1_data}')
        
            #confusion = confusion_matrix(
            #    Y_true.flatten(), Y.flatten()).astype(np.float64)
            #confusion /= np.sum(confusion)
            #print(f'confusion: {confusion}')
        
            ''' eval on KB deduction '''
            f1_kb = f1_score(Y_deduction.flatten(), Y.flatten(), average="macro")
            f1_bal = weighted_mean(f1_data, f1_kb, .5)
            #print(f'KB f1: {f1_kb}, balanced f1: {f1_bal}')
        
            data_f1_lst.append(f1_data)
            kb_f1_lst.append(f1_kb)
            bal_f1_lst.append(f1_bal)

            for k,deduction in xref_deduction.items():
                xref_f1_lst[k].append(f1_score(np.abs(deduction.flatten()), np.abs(Y.flatten())))
        
        mse_mean, mse_stde = .5*(max(mse_lst)+min(mse_lst)), .5*(max(mse_lst)-min(mse_lst))
        data_f1_mean, data_f1_stde = .5*(max(data_f1_lst)+min(data_f1_lst)), .5*(max(data_f1_lst)-min(data_f1_lst))
        kb_f1_mean, kb_f1_stde = .5*(max(kb_f1_lst)+min(kb_f1_lst)), .5*(max(kb_f1_lst)-min(kb_f1_lst))
        bal_f1_mean, bal_f1_stde = .5*(max(bal_f1_lst)+min(bal_f1_lst)), .5*(max(bal_f1_lst)-min(bal_f1_lst))

        mse_mean_lst.append(mse_mean)
        mse_stde_lst.append(mse_stde)
        data_f1_mean_lst.append(data_f1_mean)
        data_f1_stde_lst.append(data_f1_stde)
        kb_f1_mean_lst.append(kb_f1_mean)
        kb_f1_stde_lst.append(kb_f1_stde)
        bal_f1_mean_lst.append(bal_f1_mean)
        bal_f1_stde_lst.append(bal_f1_stde)

        for (k,v),mean_lst,stde_lst in zip(xref_f1_lst.items(), xref_mean_lst, xref_stde_lst):
            if len(v) > 0:
                mean, stde = .5*(max(v)+min(v)), .5*(max(v)-min(v))
                mean_lst.append(mean)
                stde_lst.append(stde)
            else:
                mean_lst.append(None)
                stde_lst.append(None)
        
        print( f'{model_name}, {data_name} data, MSE:         {mse_mean: .4f} ± {mse_stde: .4f}')
        print( f'{model_name}, {data_name} data, f1 on test:  {data_f1_mean: .4f} ± {data_f1_stde: .4f}')
        print( f'{model_name}, {data_name} data, f1 on KB:    {kb_f1_mean: .4f} ± {kb_f1_stde: .4f}')
        print( f'{model_name}, {data_name} data, balanced F1: {bal_f1_mean: .4f} ± {bal_f1_stde: .4f}')
        exit()

    res_df = pd.DataFrame({
        'data_name':    [data_name]*len(mse_mean_lst),
        'model':        model_lst,
        'mse_mean':     mse_mean_lst,
        'mse_stde':     mse_stde_lst,
        'data_f1_mean': data_f1_mean_lst,
        'data_f1_stde': data_f1_stde_lst,
        'kb_f1_mean':   kb_f1_mean_lst,
        'kb_f1_stde':   kb_f1_stde_lst,
        'bal_f1_mean':  bal_f1_mean_lst,
        'bal_f1_stde':  bal_f1_stde_lst,
        'corum_mean':   xref_mean_lst[0],
        'corum_stde':   xref_stde_lst[0],
        'string_mean':  xref_mean_lst[1],
        'string_stde':  xref_stde_lst[1],
        'chipseq_mean': xref_mean_lst[2],
        'chipseq_stde': xref_stde_lst[2],
        'lr_mean':      xref_mean_lst[3],
        'lr_stde':      xref_stde_lst[3],
        }).set_index(['data_name', 'model'])
    #res_df.to_csv(f'results/ex1_baselines/{data_name}_results.csv', index=True)
    df_lst.append(res_df)
final_df = pd.concat(df_lst, axis=0)
final_df.to_csv(f'results/ex1_baselines/benchmark_results.csv', index=True)
