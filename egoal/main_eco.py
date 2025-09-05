import torch
import pandas as pd
import numpy as np
from datetime import datetime

from egoal.abl import abduce

if __name__ == "__main__":
    log_file = f'log/EGOAL-eco-{datetime.now()}.txt'.replace(' ','-')

    model_type = 'GNN'
    model_name = model_type + '_eco_Sep3_8'
    seed = 6666
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(model_name)
    print(log_file)
    print(f'random seed: {seed}')

    label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
    idx_list_p1k = list(label_set['precise1k_idx'])
    idx_list_sra = list(label_set['matrix_idx'])


    X_p1k = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype = torch.float32)
    Y_p1k = torch.tensor(np.load('dataset/precise1k/Y_label.npy'), dtype = int)

    X_sra = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype = torch.float32)
    Y_sra = torch.tensor(np.load('dataset/ncbi-sra/Y_label.npy'), dtype = int)

    Y_p1k = Y_p1k[:,idx_list_p1k]
    Y_sra = Y_sra[:,idx_list_sra]

    test_idx_p1k = np.zeros(len(X_p1k), dtype=bool)
    test_idx_p1k[[280,281,282,283,284,285,286,287,288,289,\
            290,291,  292,293,294,295,296,297,298,299,\
            300,301,302,303,304,305,306,307,308,309,\
            310,311,312,313,314,315,316,317,318,319,\
            320,321,  322,323]] = True
            # b1109,b0734,b0978,  b2287,b0734,  b2287,b0734,b0978,
            # b1109,b0431,  b2287,b0431,  b1109,b0734,  b2287,b0734

    test_idx_sra = np.zeros(len(X_sra), dtype=bool)
    test_idx_sra[[37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]] = True
    # arcZ, gcvB, micA, ryhB

    X_test = torch.concat([X_p1k[test_idx_p1k], X_sra[test_idx_sra]])
    Y_test = torch.concat([Y_p1k[test_idx_p1k], Y_sra[test_idx_sra]])
    X_train = torch.concat([X_p1k[~test_idx_p1k], X_sra[~test_idx_sra]])
    Y_train = torch.concat([Y_p1k[~test_idx_p1k], Y_sra[~test_idx_sra]])
    print(f'train shape: {Y_train.shape}, test shape: {Y_test.shape}')

    X_unlabel = torch.tensor(np.load('dataset/X_regulators.npy'), dtype = torch.float32)
    label_weight = torch.tensor(np.load('rules/label_weight.npy'))


    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    X_unlabel = X_unlabel.to(device)
    label_weight = label_weight.to(device)

    abduce(X_unlabel= X_unlabel,
           X_test= X_test,
           Y_test= Y_test,

           pos_trn_pth='rules/regu_pos.npz',
           neg_trn_pth='rules/regu_neg.npz',
           closure = 5,
           closure_type= 'weighted',
           adj_matrix_closure= True,
           gnn_extra_layer= True,

           output_idx_list=idx_list_sra,
           label_weight=label_weight,
           weight_init_epc= 2000,
           weight_init_lr= 1e-3,

           X_label = X_train,
           Y_label = Y_train,
           #pretrained_model_pth= 'models/pretrained_7.18_label_weight.pt',
           model_save_pth = f'models/{model_name}',
           base_learner_type= model_type,

           T= 2,

           pretrain_epc= 500,
           pretrain_rl_epc= 1,
           pretrain_lr= 1e-3,

           retrain_epc= 400,
           retrain_rl_epc= 20,
           retrain_lr= 1e-3,
           refine_epc= 5000,
           refine_lr= 1e-3,

           device= device,
           seed= seed,
           log_file= log_file,
           verbose= True)
