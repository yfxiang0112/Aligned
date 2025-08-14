import torch
import numpy as np
from scipy.sparse import load_npz
from datetime import datetime

from egoal.abl import abduce

if __name__ == '__main__':
    data_name = 'norman'
    log_file = f'log/EGOAL-hsa-{datetime.now()}.txt'.replace(' ','-')
    
    model_type = 'GNN'
    model_name = model_type + '_human_Aug13'
    print(model_name)
    print(log_file)

    #########################################

    np.random.seed(42)
    X_train = torch.tensor(load_npz(f'dataset/human/{data_name}_X.npz').toarray(), dtype = torch.float32)
    Y_train = torch.tensor(load_npz(f'dataset/human/{data_name}_Y.npz').toarray(), dtype = int)

    #test_idx = np.random.choice([True, False], size=len(X_train), p=[.2, .8])
    p_train = .5
    test_idx = np.zeros(shape=len(X_train), dtype=bool)
    test_idx[np.load(f'dataset/human/{data_name}_test_idx.npy')] = True

    train_idx = np.random.choice([True, False], size=len(X_train)-np.count_nonzero(test_idx), p=[p_train, 1-p_train])

    X_test = X_train[test_idx]
    Y_test = Y_train[test_idx]
    X_train = X_train[~ test_idx][train_idx]
    Y_train = Y_train[~ test_idx][train_idx]

    regulators = np.nonzero(np.sum(\
            load_npz(f'rules/human/{data_name}_KB_P.npz').toarray()\
            +load_npz(f'rules/human/{data_name}_KB_N.npz').toarray(), axis=1))[0]
    X_unlabel = np.zeros(shape=(len(regulators), X_train.shape[1]))
    X_unlabel[range(len(X_unlabel)), regulators] = 1.
    X_unlabel = torch.tensor(X_unlabel, dtype=torch.float32)
    print('unlabel:', X_unlabel.shape)

    label_weight = torch.tensor(np.load(f'dataset/human/{data_name}_label_weight.npy'))

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    X_unlabel = X_unlabel.to(device)
    label_weight = label_weight.to(device)

    abduce(X_unlabel= X_unlabel,
           X_test= X_test,
           Y_test= Y_test,

           pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
           neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
           label_weight=label_weight,

           X_label = X_train,
           Y_label = Y_train,
           #pretrained_model_pth = '',
           model_save_pth = f'models/{model_name}',
           base_learner_type= model_type,

           T= 2,

           pretrain_epc= 300,
           pretrain_rl_epc= 1,
           pretrain_lr= 1e-3,

           retrain_epc= 400,
           retrain_rl_epc= 100,
           retrain_lr= 1e-3,
           refine_epc= 0,
           refine_lr= 1e-4,

           device= device,
           seed= 42,
           log_file= log_file,
           verbose= True)
