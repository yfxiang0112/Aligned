import torch
import pandas as pd
import numpy as np
from datetime import datetime

from egoal.abl import abduce

if __name__ == "__main__":
    log_file = f'log/EGOAL-eco-{datetime.now()}.txt'.replace(' ','-')

    X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype = torch.float32)
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_label.npy'), dtype = int)

    X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype = torch.float32)
    Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_label.npy'), dtype = int)

    test_idx = [37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]
    # arcZ, gcvB, micA, ryhB
    X_test, Y_test = X_test[test_idx], Y_test[test_idx]

    X_unlabel = torch.tensor(np.load('dataset/X_regulators.npy'), dtype = torch.float32)
    #X_unlabel = X_train

    label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
    idx_list_p1k = list(label_set['precise1k_idx'])
    idx_list_sra = list(label_set['matrix_idx'])

    Y_train = Y_train[:,idx_list_p1k]
    Y_test = Y_test[:,idx_list_sra]

    label_weight = torch.tensor(np.load('rules/label_weight.npy'))


    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    X_unlabel = X_unlabel.to(device)
    label_weight = label_weight.to(device)

    abduce(X_unlabel= X_unlabel,
           X_test= X_test,
           Y_test= Y_test,

           pos_trn_pth='rules/regu_pos.npz',
           neg_trn_pth='rules/regu_neg.npz',
           closure_type= 'weighted',
           output_idx_list=idx_list_sra,
           label_weight=label_weight,

           X_label = X_train,
           Y_label = Y_train,
           #pretrained_model_pth= 'models/pretrained_7.18_label_weight.pt',
           #model_save_pth= 'models/ecoli/MLP_Aug20.pt',
           base_learner_type= 'GNN',

           T= 2,

           pretrain_epc= 500,
           pretrain_rl_epc= 1,
           pretrain_lr= 1e-3,

           retrain_epc= 400,
           retrain_rl_epc= 100,
           retrain_lr= 1e-4,
           refine_epc= 5000,
           refine_lr= 1e-3,

           device= device,
           seed= 42,
           log_file= log_file,
           verbose= True)
