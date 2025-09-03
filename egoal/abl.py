import numpy as np
import pandas as pd
import torch
import cupy as cp
from tqdm import tqdm
from datetime import datetime
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt, parameter
from sklearn.metrics import f1_score

from egoal.learner_refl import ReflectLearner
from egoal.reasoner import RegulatoryKB#, MetabolicKB

def eval_weight(X: torch.Tensor, Y: torch.Tensor, KB: RegulatoryKB) -> float:
    '''
    Compute the weight in integrated data-knowledge evaluation metric
    Args:
        X:
        Y:
        KB:
    Return:
        w_data
    '''
    size_y = int(Y.shape[0]*Y.shape[1])
    size_data= int(torch.count_nonzero(torch.sum(Y, dim=1)))
    size_klg= .5* (int(torch.count_nonzero(torch.sum(KB.KB, dim=1)))\
                    +int(torch.count_nonzero(torch.sum(KB.KB, dim=0))))

    Y_deduction = KB.deduce(X)

    q_data = (int(torch.count_nonzero(Y)) / size_y) +\
            (size_data/(size_klg+size_data)) +\
            (int(torch.count_nonzero((torch.sum(Y, dim=1)!=0) & (torch.sum(Y_deduction,dim=1)==0))) / len(Y))

    q_knowledge = (int(torch.count_nonzero(Y_deduction)) / size_y) +\
            (size_klg/(size_klg+size_data)) +\
            (int(torch.count_nonzero((torch.sum(Y, dim=1)==0) & (torch.sum(Y_deduction,dim=1)!=0))) / len(Y))

    return q_knowledge / (q_data + q_knowledge)

# TODO
def abduce(X_unlabel: torch.Tensor,
           X_test: torch.Tensor,
           Y_test: torch.Tensor,
           X_label: torch.Tensor,
           Y_label: torch.Tensor,

           pos_trn_pth: str,
           neg_trn_pth: str | None,
           closure_type = 'naive',
           closure= 5,

           output_idx_list = None,
           label_weight = None | torch.Tensor,
           weight_init_epc = 1000,
           weight_init_lr = 1e-4,

           pretrained_model_pth = None,
           model_save_pth = None,
           base_learner_type = 'MLP',
           adj_matrix_closure = False,
           gnn_extra_layer = False,

           T= 5,
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
        X_label:
        Y_label:

        pos_trn_pth: str:
        neg_trn_pth: str:
        output_idx_list = None:

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
    reasoner = RegulatoryKB(pos_trn_pth= pos_trn_pth,
                            neg_trn_pth= neg_trn_pth,
                            output_idx_list= output_idx_list,
                            device=device)#, T=4)
    reasoner.closure_(T=closure, closure_type=closure_type)

    if base_learner_type == 'GNN':
        adj_matrix = torch.round(torch.abs(reasoner.KB)) if adj_matrix_closure\
                else torch.round(torch.clamp(torch.abs(reasoner.Regu_P_0 + reasoner.Regu_N_0), 0,1))
    else:
        adj_matrix = None

    learner = ReflectLearner(input_dim= X_test.shape[1],
                             output_dim= Y_test.shape[1],
                             hidden_dim= 64,
                             base_learner_type= base_learner_type,
                             adj_matrix= adj_matrix,
                             gnn_extra_layer= gnn_extra_layer,
                             device= device,
                             log_path= log_file)
    if label_weight != None:
        learner.init_weight(label_weight, epochs=weight_init_epc, lr=weight_init_lr)

    ########################################

    ''' base learner training (or loading) '''
    if pretrained_model_pth != None:
        learner.load_data(None, None, X_test, Y_test)

        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nbefore pretrain\n{"-"*20}\n')
        w_data = eval_weight(X_label, Y_label, reasoner)
        print(f'Eval weight w_data: {w_data}')
        f1 = learner.eval(reasoner, w_data)
        print(f'Before pretrain: integrated f1 {f1:.4f}')

        learner.load(pretrained_model_pth)

    elif X_label != None and Y_label != None:
        learner.load_data(X_label, Y_label, X_test, Y_test)

        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nbefore pretrain\n{"-"*20}\n')
        w_data = eval_weight(X_label, Y_label, reasoner)
        print(f'Eval weight w_data: {w_data}')
        f1 = learner.eval(reasoner, w_data)
        print(f'Before pretrain: integrated f1 {f1:.4f}')

        learner.train(KB= reasoner,
                      label_weight= label_weight,
                      epochs= pretrain_epc,
                      reinforce_epochs= pretrain_rl_epc,
                      C=10,
                      lr=pretrain_lr,
                      verbose=verbose)
        learner.save('models/pretrained.pt' if model_save_pth==None else model_save_pth+'.pt')

    else:
        learner.load_data(None, None, X_test, Y_test)


    if log_file != '':
        with open(log_file, 'a') as log:
            log.write(f'\n\nbefore ABL\n{"-"*20}\n')

    print('------ Before ABL ------')
    w_data = eval_weight(X_label, Y_label, reasoner)
    f1 = learner.eval(reasoner, w_data, verbose=True)
    print(f'integrated f1 {f1:.4f}\n')

    ########################################

    ''' abl main loop '''
    for t in range(T):
        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nABL loop {t+1}\n{"-"*20}\n')

        ' predict pseudo label & to binary '
        Y_prob, R = learner.forward(X_unlabel)
        Y_pseudo = torch.argmax(Y_prob, dim=-1) -1

        Y_deduction = reasoner.deduce(X_unlabel)

        ' retrain base learner '
        Y_modified = torch.where(R > .5, Y_deduction, Y_pseudo)
        learner.load_data(X_unlabel, Y_modified, X_test, Y_test, update_weight=True)
        learner.train(KB= reasoner,
                      label_weight= label_weight,
                      epochs= retrain_epc,
                      reinforce_epochs= retrain_rl_epc,
                      C= 100,
                      lr= retrain_lr,
                      verbose= verbose)
        learner.save(f'models/ABL_{t}.pt' if model_save_pth==None else model_save_pth+f'_ABL_{t}.pt')

        print(f'------ ABL Loop {t} ------')
        #w_data = eval_weight(X_label, Y_label, reasoner)
        f1 = learner.eval(reasoner, w_data, verbose=True)
        print(f'integrated f1 {f1:.4f}')

        ' knowledge refine '
        reasoner.refine(X= X_unlabel,
                        Y= Y_modified,
                        k= closure,
                        epochs= refine_epc,
                        lr= refine_lr,
                        approx= 'tanh',
                        verbose= verbose)

        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\nafter refine:\n')
        print(f'------ ABL {t} after refine ------')
        #w_data = eval_weight(X_label, Y_label, reasoner)
        f1 = learner.eval(reasoner, w_data, verbose=True)
        print(f'integrated f1 {f1:.4f}')
