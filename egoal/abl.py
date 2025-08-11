import numpy as np
import pandas as pd
import torch
import cupy as cp
from tqdm import tqdm
from datetime import datetime
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt, parameter
from sklearn.metrics import f1_score

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
    reasoner = RegualtoryKB(pos_trn_pth= pos_trn_pth,
                            neg_trn_pth= neg_trn_pth,
                            output_idx_list= output_idx_list,
                            device=device)#, T=4)
    reasoner.closure_(T=closure, closure_type=closure_type)

    if base_learner_type == 'GNN':
        adj_matrix = torch.round(torch.abs(reasoner.get_KB()))
    else:
        adj_matrix = None

    learner = ReflectLearner(input_dim= X_test.shape[1],
                             output_dim= Y_test.shape[1],
                             hidden_dim= 64,
                             base_learner_type= base_learner_type,
                             adj_matrix= adj_matrix,
                             device=device,
                             log_path=log_file)
    if label_weight != None:
        learner.init_weight(label_weight)

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
        R_binary = R >= .6#torch.round(R > .8).bool()
        print('R_biary nonzero:', torch.count_nonzero(R_binary))


        print(torch.count_nonzero(R_binary) / (R.shape[0]*R.shape[1]))
        n_rows, n_cols = Y_pseudo.shape[0], Y_pseudo.shape[1]

        Y_deduction = reasoner.deduce(X_unlabel)

        Y_modified = torch.where(R_binary, Y_deduction, Y_pseudo)
        #Y_modified = Y_deduction


        #NOTE tmp ########################################
        Y_pred, R_pred = learner.forward(X_test)
        Y_pred = torch.argmax(Y_pred, dim=-1) -1
        y_p_flat = Y_pred.detach().cpu().numpy().flatten()
        y_t_flat = Y_test.detach().cpu().numpy().flatten()
        print(f'Y_pseudo f1: {f1_score(y_t_flat, y_p_flat, average="macro")}')
        print(f'Y_pseudo f1 (abs): {f1_score(np.abs(y_t_flat), np.abs(y_p_flat), average="macro")}')

        R_pred = R_pred >= .5 #NOTE
        Y_deduction_test = reasoner.deduce(X_test)

        y_d_flat = Y_deduction_test.detach().cpu().numpy().flatten()
        print(f'Y_pseudo f1 (Y_d): {f1_score(y_d_flat, y_p_flat, average="macro")}')

        #np.save(f'data_anal/abduction_results/R_ABL{t}_hsa.npy', R_pred.cpu().numpy()) #NOTE tmp
        #np.save(f'data_anal/abduction_results/Yp_ABL{t}_hsa.npy', Y_pred.cpu().numpy()) #NOTE tmp
        #np.save(f'data_anal/abduction_results/Yd_ABL{t}_hsa.npy', Y_deduction_test.cpu().numpy()) #NOTE tmp

        Y_pred = torch.where(R_pred, Y_deduction_test, Y_pred)

        y_p_flat = Y_pred.detach().cpu().numpy().flatten()
        #y_t_flat = Y_test.detach().cpu().numpy().flatten()
        #print(Y_modified.shape, Y_test.shape)
        print(f'Y_modified f1: {f1_score(y_t_flat, y_p_flat, average="macro")}')
        print(f'Y_modified f1 (abs): {f1_score(np.abs(y_t_flat), np.abs(y_p_flat), average="macro")}')
        print(f'Y_modified f1 (Y_d): {f1_score(y_d_flat, y_p_flat, average="macro")}')

        #exit()
        #if t == 1: # tmp
        #    exit()

        #NOTE end #######################################

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
        learner.save(f'models/ABL_{t}.pt' if model_save_pth==None else model_save_pth+f'ABL_{t}')

        f1 = learner.eval()
        print(f'ABL loop {t}: macro f1 {f1:.4f}')
