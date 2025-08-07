import numpy as np
import pandas as pd
import torch
import cupy as cp
from tqdm import tqdm
from datetime import datetime
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt, parameter
from sklearn.metrics import f1_score

from egoal.learner import BaseLearner
from egoal.learner_refl import ReflectLearner
from egoal.reasoner import RegualtoryKB#, MetabolicKB
from egoal.utils import optvec2matrix

# TODO
def abduce(X_unlabel: torch.Tensor,
           X_test: torch.Tensor,
           Y_test: torch.Tensor,

           pos_trn_pth: str,
           neg_trn_pth: str | None,
           closure_type = 'naive',
           output_idx_list = None,
           label_weight = None | torch.Tensor,

           X_label = None | torch.Tensor,
           Y_label = None | torch.Tensor,
           pretrained_model_pth = None,
           model_save_pth = None,
           base_learner_type = 'MLP',

           T= 5,

           closure= 5,
           pretrain_epc= 300,
           pretrain_rl_epc= 100,
           pretrain_lr= 1e-3,

           retrain_epc= 500,
           retrain_lr= 1e-3,
           retrain_rl_epc= 10,
           refine_epc= 2000,
           refine_lr= 1e-4,

           device= 'cpu',
           seed= None,
           log_file= '',
           verbose= False):
    '''
    Abductive Learning Main Loop

    Args:
        X_unlabel: torch.Tensor:
        X_test: torch.Tensor:
        Y_test: torch.Tensor:

        pos_trn_pth: str:
        neg_trn_pth: str:
        output_idx_list = None:

        X_label:
        Y_label:
        pretrained_model_pth:
        model_save_pth:
        base_learner_type:

        T:
        pretrain_epc:
        pretrain_lr:
        retrain_epc:
        retrain_lr:
        device:
        seed:
        log_file:
    '''

    
    if seed != None:
       torch.manual_seed(seed)
       np.random.seed(seed)

    ''' init base learner & reasoner  '''
    learner = ReflectLearner(input_dim= X_test.shape[1],
                             output_dim= Y_test.shape[1],
                             hidden_dim= 64,
                             base_learner_type= base_learner_type,
                             device=device,
                             log_path=log_file)
    reasoner = RegualtoryKB(pos_trn_pth= pos_trn_pth,
                            neg_trn_pth= neg_trn_pth,
                            output_idx_list= output_idx_list,
                            device=device)#, T=4)
    reasoner.closure_(T=closure, closure_type=closure_type)

    ########################################

    ''' base learner training (or loading) '''
    if pretrained_model_pth != None:
        learner.load_data(None, None, X_test, Y_test)

        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nbefore pretrain\n{"-"*20}\n')
        f1 = learner.eval()
        print(f'Before pretrain: macro f1 {f1:.4f}')

        learner.load(pretrained_model_pth)

    elif X_label != None and Y_label != None:
        learner.load_data(X_label, Y_label, X_test, Y_test)

        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nbefore pretrain\n{"-"*20}\n')
        f1 = learner.eval()
        print(f'Before pretrain: macro f1 {f1:.4f}')

        learner.train(KB= reasoner,
                      label_weight= label_weight,
                      epochs= pretrain_epc,
                      reinforce_epochs= pretrain_rl_epc,
                      C=10,
                      lr=pretrain_lr,
                      verbose=verbose)
        learner.save('models/pretrained.pt' if model_save_pth==None else model_save_pth)

    else:
        learner.load_data(None, None, X_test, Y_test)


    if log_file != '':
        with open(log_file, 'a') as log:
            log.write(f'\n\nbefore ABL\n{"-"*20}\n')
    f1 = learner.eval()
    print(f'Before ABL: macro f1 {f1:.4f}')


    ########################################

    ' modified labels in each unlabeled sample '
    modified_labels = [set() for _ in range(len(X_unlabel))]
    #TODO?

    ''' abl main loop '''
    for t in range(T):
        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nABL loop {t+1}\n{"-"*20}\n')

        ' predict pseudo label & to binary '
        Y_prob, R = learner.forward(X_unlabel)
        Y_pseudo = torch.argmax(Y_prob, dim=-1) -1
        print(torch.max(R))
        print('R >.9:', torch.sum(R > .9) / (R.shape[0]*R.shape[1]))
        print('R >.8:', torch.sum((R > .8) & (R <= .9)) / (R.shape[0]*R.shape[1]))
        print('R >.5:', torch.sum((R > .5) & (R <= .8)) / (R.shape[0]*R.shape[1]))
        print('R <.5:', torch.sum(R <= .5) / (R.shape[0]*R.shape[1]))
        R_binary = R >= .8#torch.round(R > .8).bool()
        print('R_biary nonzero:', torch.count_nonzero(R_binary))


        print(torch.count_nonzero(R_binary) / (R.shape[0]*R.shape[1]))
        n_rows, n_cols = Y_pseudo.shape[0], Y_pseudo.shape[1]

        Y_deduction = reasoner.deduce(X_unlabel)

        Y_modified = torch.where(R_binary, Y_deduction, Y_pseudo)
        #Y_modified = Y_deduction

        #print(torch.count_nonzero(Y_deduction))

        #NOTE tmp ########################################
        Y_pred, R_pred = learner.forward(X_test)
        Y_pred = torch.argmax(Y_pred, dim=-1) -1
        R_pred = R_pred >= .8
        Y_deduction_test = reasoner.deduce(X_test)

        np.save(f'data_anal/abduction_results/R_ABL{t}_hsa.npy', R_pred.cpu().numpy()) #NOTE tmp
        np.save(f'data_anal/abduction_results/Yp_ABL{t}_hsa.npy', Y_pred.cpu().numpy()) #NOTE tmp
        np.save(f'data_anal/abduction_results/Yd_ABL{t}_hsa.npy', Y_deduction_test.cpu().numpy()) #NOTE tmp

        Y_pred = torch.where(R_pred, Y_deduction_test, Y_pred)

        y_p_flat = Y_pred.detach().cpu().numpy().flatten()
        y_t_flat = Y_test.detach().cpu().numpy().flatten()
        #print(Y_modified.shape, Y_test.shape)
        print(f'Y_modified f1: {f1_score(y_t_flat, y_p_flat, average="macro")}')
        #if t == 1: # tmp
        #    exit()
        #NOTE ###########################################

        # TODO KB update before RL training?
        #np.save('KB_before.npy', reasoner.KB.detach().cpu().numpy()) #NOTE tmp

        reasoner.refine(X= X_unlabel,
                        Y= Y_modified,
                        k= closure,
                        epochs= refine_epc,
                        init_lr= refine_lr,
                        verbose= verbose)
        
        #np.save('KB_after.npy', reasoner.KB.detach().cpu().numpy()) #NOTE tmp

        ' retrain base learner '
        learner.load_data(X_unlabel, Y_modified, X_test, Y_test, update_weight=True)
        learner.train(KB= reasoner,
                      label_weight= label_weight,
                      epochs= retrain_epc,
                      reinforce_epochs= retrain_rl_epc,
                      C= 100,
                      lr= retrain_lr,
                      verbose= verbose)

        f1 = learner.eval()
        print(f'ABL loop {t}: macro f1 {f1:.4f}')
        


####################

if __name__ == "__main__":
    log_file = f'log/EGOAL-{datetime.now()}.txt'.replace(' ','-')

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

    #print(f'Y_train: {Y_train.shape}, Y_test: {Y_test.shape}')

    ##X_unlabel = X_unlabel[2860:2880] # yicR
    ##X_unlabel = X_unlabel[2570:2580]

    #X_unlabel = torch.zeros(size=(10, X_train.shape[1]), dtype=torch.float32)
    #for i in range(10):
    #    X_unlabel[i, 3370+i] = 1 #arcZ
    #    #X_unlabel[i, 2830+i] = 1 #micA
    #    #X_unlabel[i,2960+i] = 1 # gcvB

    #print(list(torch.nonzero(X_unlabel)))


    ##idx_list = [26,27] # yicR
    ##idx_list = [51,52,53,54,55,56] #gcvB
    ##idx_list = [0,1,2,4,5,11,12] # mazF

    #idx_list = list(range(42,54)) # arcZ
    ##idx_list = list(range(60,63)) # micA

    #print(torch.nonzero(X_test[idx_list]).tolist())
    #print(torch.nonzero(X_test[[i for i in range(len(X_test)) if i not in idx_list]]).tolist())
    #X_test = X_test[idx_list]
    #Y_test = Y_test[idx_list]


    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
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
           #pretrained_model_pth= 'models/pretrained_7.29_GNN.pt',
           #model_save_pth= 'models/pretrained_8.7_GNN.pt',
           base_learner_type= 'GNN',

           T= 2,

           pretrain_epc= 300,
           pretrain_rl_epc= 100,
           pretrain_lr= 1e-3,

           retrain_epc= 500,
           retrain_rl_epc= 100,
           retrain_lr= 1e-3,
           refine_epc= 2000,
           refine_lr= 1e-4,

           device= device,
           seed= 42,
           log_file= log_file,
           verbose= True)
