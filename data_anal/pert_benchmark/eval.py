import pandas as pd
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import load_npz
import glob
import os

from egoal.reasoner import RegulatoryKB

def weighted_mean(f1_data,f1_kb,w):
    p_integrate = 2
    return (w * (f1_data ** -p_integrate)\
            + (1.-w) * (f1_kb ** -p_integrate)) ** (-1/p_integrate)

data_name = 'dixit'
model_name = 'additive'

test_metadata = pd.read_csv(f'dataset/human/{data_name}_test_set.csv', index_col=0)

directory = f'data_anal/pert_benchmark/{model_name}_{data_name}'
file_pattern = 'all_predictions*.json'
log_files = glob.glob(os.path.join(directory, file_pattern))
log_files.sort()  # Sort for consistent ordering

if not log_files:
    assert 0

data_f1_lst, kb_f1_lst, bal_f1_lst = [],[],[]

for file_idx, log_file in enumerate(log_files, 1):
    print(file_idx, log_file)
    pred_result = json.load(open(log_file, 'r'))

    genome_size = len(list(pred_result.values())[0])
    
    test_idx = np.load(f'dataset/human/{data_name}_test_idx.npy')
    Y_true = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()[test_idx]
    X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()[test_idx]
    
    indices = list(test_metadata.apply(lambda x: (x['data_start_idx'], x['data_end_idx+1']), axis=1))
    pert_keys = test_metadata['pert'].apply(lambda x: f'{eval(x)[0]}_{eval(x)[1]}' if 'ctrl' not in x else list(set(eval(x))-{'ctrl'})[0])
    predictions = list(pert_keys.apply(lambda x: pred_result[x] if x in pred_result else [0]*genome_size))
    
    
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
    
    ' eval MSE '
    Y_true_con = load_npz(f'dataset/human/{data_name}_Y_con.npz').toarray()[test_idx]
    criterion = torch.nn.MSELoss(reduction='mean')
    print(f'MSE: {criterion(torch.tensor(Y), torch.tensor(Y_true_con))}')
    
    
    Y = np.where(np.abs(Y)>.1, np.sign(Y), 0)
    
    ''' eval data consistency '''
    print(Y.shape)
    
    f1_data = f1_score(Y_true.flatten(), Y.flatten(), average="macro")
    print(f'data f1: {f1_data}')
    
    confusion = confusion_matrix(Y_true.flatten(), Y.flatten()).astype(np.float64)
    confusion /= np.sum(confusion)
    print(f'confusion: {confusion}')
    
    ''' eval on KB deduction '''
    device = 'cuda'
    KB = RegulatoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz', neg_trn_pth=f'rules/human/{data_name}_KB_N.npz', device=device)
    KB.closure_(T=5, closure_type='weighted')
    Y_deduction = KB.deduce(torch.tensor(X).float().to(device)).to('cpu').numpy()
    
    f1_kb = f1_score(Y_deduction.flatten(), Y.flatten(), average="macro")
    f1_bal = weighted_mean(f1_data, f1_kb, .3233)
    print(f'KB f1: {f1_kb}, balanced f1: {f1_bal}')
    
    data_f1_lst.append(f1_data)
    kb_f1_lst.append(f1_kb)
    bal_f1_lst.append(f1_bal)

print(f'data f1: {.5*(max(data_f1_lst)+min(data_f1_lst)): .4f} \\pm {.5*(max(data_f1_lst)-min(data_f1_lst)): .4f}')
print(f'kb f1: {.5*(max(kb_f1_lst)+min(kb_f1_lst)): .4f} \\pm {.5*(max(kb_f1_lst)-min(kb_f1_lst)): .4f}')
print(f'balanced f1: {.5*(max(bal_f1_lst)+min(bal_f1_lst)): .4f} \\pm {.5*(max(bal_f1_lst)-min(bal_f1_lst)): .4f}')
