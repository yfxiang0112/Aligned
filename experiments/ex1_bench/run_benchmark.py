import torch
import numpy as np
from scipy.sparse import load_npz
from datetime import datetime
import argparse
import pandas as pd

from aligned.abl import abduce

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', dest= 'data_name', action= 'store', required= True, type=str)
    parser.add_argument('--log_file', dest= 'log_file', action= 'store', default= f'log/EGOAL-hsa-{datetime.now()}.txt'.replace(' ','-'), type=str)
    parser.add_argument('--model_type', dest= 'model_type', action= 'store', default= 'GNN', type=str)
    parser.add_argument('--model_save_name', dest= 'model_name', action= 'store', default= '', type=str)
    parser.add_argument('--seed', dest= 'seed', action= 'store', default= 42, type=int)
    parser.add_argument('--device', dest= 'device', action= 'store', default= 'cuda', type=str)
    parser.add_argument('--trainset_remove', dest= 'p_train', action= 'store', default= 1., type=float)
    parser.add_argument('--random_split', dest= 'random_split', action= 'store', default= False, type=bool)
    args = parser.parse_args()

    data_name = args.data_name
    log_file = args.log_file
    model_type = args.model_type
    model_name = f'{model_type}_{data_name}_{args.model_name}'
    seed = args.seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(model_name)
    print(log_file)
    print(f'random seed: {seed}')

    #########################################

    np.random.seed(seed)
    X_train = torch.tensor(load_npz(f'dataset/human/{data_name}_X.npz').toarray(), dtype = torch.float32)
    Y_train = torch.tensor(load_npz(f'dataset/human/{data_name}_Y.npz').toarray(), dtype = int)

    #test_idx = np.random.choice([True, False], size=len(X_train), p=[.3, .7])
    test_idx = np.zeros(shape=len(X_train), dtype=bool)
    if not args.random_split:
        test_idx[np.load(f'dataset/human/{data_name}_test_idx.npy')] = True
    else:
        metadata = pd.read_csv(f'dataset/human/{data_name}_metadata.csv',index_col=0)
        test_pert = np.random.choice([True, False], size=len(metadata), p=[.2, .8])\
                & (metadata['pert'].apply(lambda x: len(eval(x))>1))
        print(f'test perts: {list(metadata.loc[test_pert, "pert"])}')
        test_data_idx = sum(metadata[test_pert]
                .apply(lambda x: list(range(x['data_start_idx'],x['data_end_idx+1'])), axis=1), [])
        test_idx[test_data_idx] = True
    np.save(f'models/{model_name}_split.npy', test_idx)

    p_train = args.p_train
    train_idx = np.random.choice([True, False], size=len(X_train)-np.count_nonzero(test_idx), p=[p_train, 1-p_train])

    X_test = X_train[test_idx]
    Y_test = Y_train[test_idx]
    X_train = X_train[~ test_idx][train_idx]
    Y_train = Y_train[~ test_idx][train_idx]
    print('train:', X_train.shape)

    regulators = np.nonzero(np.sum(\
            load_npz(f'rules/human/{data_name}_KB_P.npz').toarray()\
            +load_npz(f'rules/human/{data_name}_KB_N.npz').toarray(), axis=1))[0]
            #load_npz(f'scripts/test/rand_kb/{data_name}_KB_P_rand.npz').toarray()\
            #+load_npz(f'scripts/test/rand_kb/{data_name}_KB_N_rand.npz').toarray(), axis=1))[0]
    X_unlabel = np.zeros(shape=(len(regulators), X_train.shape[1]))
    X_unlabel[range(len(X_unlabel)), regulators] = 1.
    X_unlabel = torch.tensor(X_unlabel, dtype=torch.float32)
    print('unlabel:', X_unlabel.shape)

    label_weight = torch.tensor(np.load(f'dataset/human/{data_name}_label_weight.npy'))

    #label_weight = torch.zeros(Y_train.shape[1])+.5
    #label_weight = torch.tensor(np.random.uniform(.2,.8,Y_train.shape[1])).float()
    #label_weight = torch.tensor(np.load(f'scripts/test/rand_kb/{data_name}_label_weight.npy'))

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    X_unlabel = X_unlabel.to(device)
    label_weight = label_weight.to(device)

    abduce(X_unlabel= X_unlabel,
           X_test= X_test,
           Y_test= Y_test,
           X_label = X_train,
           Y_label = Y_train,

           pos_trn_pth=f'rules/human/{data_name}_KB_P.npz',
           neg_trn_pth=f'rules/human/{data_name}_KB_N.npz',
           #pos_trn_pth=f'scripts/test/rand_kb/{data_name}_KB_P_rand.npz',
           #neg_trn_pth=f'scripts/test/rand_kb/{data_name}_KB_N_rand.npz',
           #pos_trn_pth=f'data_anal/xref_csbench/recovery_KB/{data_name}_KB_P_42.npz',
           #neg_trn_pth=f'data_anal/xref_csbench/recovery_KB/{data_name}_KB_N_42.npz',
           closure = 5,
           closure_type = 'weighted',

           label_weight=label_weight,
           #pretrained_model_pth = 'models/MLP_human_Aug19.pt',
           model_save_pth = f'models/{model_name}',
           base_learner_type= model_type,

           T= 2,

           pretrain_epc= 300,
           pretrain_rl_epc= 1,
           pretrain_lr= 1e-3,

           retrain_epc= 150,
           retrain_rl_epc= 100,
           retrain_lr= 1e-3,
           refine_epc= 5000,
           refine_lr= 1e-3,

           device= device,
           seed= seed,
           log_file= log_file,
           verbose= True)
