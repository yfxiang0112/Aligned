import numpy as np
import pandas as pd
import torch
import cupy as cp
from tqdm import tqdm
from datetime import datetime
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt, parameter

from egoal.learner import BaseLearner
from egoal.learner_refl import ReflectLearner
from egoal.reasoner import RegualtoryKB#, MetabolicKB
#from egoal.refl import Reflector
from egoal.utils import optvec2matrix
#from egoal.utils import xor, negation, matrix2pgb, pgb2ternary, optvec2pgb

# TODO
def abduce(X_unlabel: torch.Tensor,
           X_test: torch.Tensor,
           Y_test: torch.Tensor,
           pos_trn_pth: str,
           neg_trn_pth: str,
           X_label = None | torch.Tensor,
           Y_label = None | torch.Tensor,
           use_gpu = False,
           T = 10, max_modify=1, budget=100, pretrain_epc=100, pretrain_lr=0.01,
           subset_threshold=0.95, retrain_epc=20, retrain_lr=0.01,
           test_idx=[], test_size=0.2, kb_weight=[],
           seed=None, log_file=''):

    if seed != None:
       torch.manual_seed(seed)
       np.random.seed(seed)


    ''' init base learner & reasoner  '''
    learner = ReflectLearner(input_dim= X_test.shape[1],
                             output_dim= Y_test.shape[1],
                             hidden_dim= 64,
                             use_gpu=use_gpu,
                             log_path=log_file)
    reasoner = RegualtoryKB(pos_trn_pth= pos_trn_pth,
                            neg_trn_pth= neg_trn_pth,
                            use_gpu=use_gpu)#, T=4)
    reasoner.closure_()

    ########################################

    ''' base learner training '''
    if X_label != None and Y_label != None:
        learner.load_data(X_label, Y_label, X_test, Y_test)

        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nbefore pretrain\n{"-"*20}\n')
        f1 = learner.eval()
        print(f'Before pretrain: macro f1 {f1:.4f}')

        learner.train(KB=reasoner, epochs=pretrain_epc, lr=pretrain_lr)

        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nafter pretrain\n{"-"*20}\n')
        f1 = learner.eval()
        print(f'After pretrain: macro f1 {f1:.4f}')
    
    ########################################

    ' modified labels in each unlabeled sample '
    modified_labels = [set() for _ in range(len(X_unlabel))]

    ''' abl main loop '''
    for t in range(T):
        if log_file != '':
            with open(log_file, 'a') as log:
                log.write(f'\n\nABL loop {t+1}\n{"-"*20}\n')

        ' predict pseudo label & to binary '
        Y_prob, R = learner.forward(X_unlabel)
        Y_pseudo = torch.argmax(Y_prob, dim=-1) -1
        R_binary = torch.round(R).bool()
        #print(R)
        #print(R_binary)
        print(torch.count_nonzero(R_binary) / (R.shape[0]*R.shape[1]))
        #exit()
        n_rows, n_cols = Y_pseudo.shape[0], Y_pseudo.shape[1]

        Y_deduction = reasoner.deduce(X_unlabel)

        Y_modified = torch.where(R_binary, Y_deduction, Y_pseudo)

        np.save(f'R_ABL{t}.npy', R_binary.cpu().numpy())
        np.save(f'Y_ABL{t}.npy', (Y_pseudo - Y_deduction).cpu().numpy())

        ####################

        #Y_modified = []
        #n_rows = len(X_unlabel)
        #mask = torch.zeros_like(Y_pseudo)
        #for idx, x_unlabel, y_pseudo, y_prob in tqdm(zip(range(n_rows),
        #                                            X_unlabel, Y_pseudo, Y_prob),
        #                                            total=n_rows, desc=f'ABL loop {t+1}'):



        #    y_neg = cp.asarray(y_pseudo==-1).astype(cp.bool_)

        #    ' select subset with prob(y_p|x) '
        #    # NOTE. temp suspended
        #    #if type(subset_threshold)==float:
        #    #    subset = (y_prob.max(dim=-1).values <= subset_threshold)
        #    #elif type(subset_threshold)==list and len(subset_threshold)>=T:
        #    #    subset = (y_prob.max(dim=-1).values <= subset_threshold[t])
        #    #else:
        #    #    subset = (y_prob.max(dim=-1).values <= 1)
        #
        #    #subset_idx = torch.nonzero(subset).squeeze().tolist()
        #    #if type(subset_idx) == int:
        #    #    subset_idx = [subset_idx]
        #    # NOTE end

        #    subset_idx = list(range(n_cols))

        #    ' select inconsistent subset w.r.t TRN '
        #    trn_deduce_p, trn_deduce_n = TRN_reasoner.deduce(x_u)
        #    incons_labels_p = set(np.nonzero(trn_deduce_p ^ y_pos)[0].tolist())
        #    incons_labels_n = set(np.nonzero(trn_deduce_n ^ y_neg)[0].tolist())
        #    dual_labels = set(np.nonzero(trn_deduce_p & trn_deduce_n)[0].tolist())
        #    incons_labels = dual_labels.union(incons_labels_p).union(incons_labels_n)

        #    subset_idx = [x for x in subset_idx if x not in modified_labels[idx] and\
        #                                 x in incons_labels]

        #    ' select subset with GEM '
        #    #TODO
        #    if kb_weight[t][1] == 1:
        #        subset_idx = [24, 25, 27, 54, 58, 74, 79, 80, 81, 104, 133, 137, 145, 160, 168, 173, 176, 191, 229, 273, 283, 286, 302, 303, 316, 367, 370, 371, 372, 389, 396, 420, 450, 460, 461, 463, 464, 512, 513, 515, 521, 563, 594, 617]
        #        #subset_idx = [0, 1, 24, 25, 27, 29, 54, 58, 74, 79, 80, 81, 94, 103, 104, 111, 113, 117, 133, 137, 139, 145, 160, 168, 173, 176, 182, 185, 186, 191, 229, 245, 272, 273, 283, 286, 302, 303, 316, 340, 350, 367, 370, 371, 372, 381, 389, 391, 396, 418, 420, 450, 460, 461, 463, 464, 466, 469, 481, 512, 513, 515, 517, 518, 521, 527, 537, 541, 561, 562, 563, 564, 569, 587, 592, 594, 604, 617]

        #    n_subset = len(subset_idx)
        #    if n_subset == 0:
        #        Y_modified.append(y_pseudo)
        #        continue

        #    ###############################################

        #    ' zoopt '
    
        #    ''' restrict search space:
        #        v in [0,|G|]^(n_unlabels * max_modify)
        #        (gene idx to modify for each sample.) '''
        #    #opt_dim = Dimension(size= n_rows * max_modify * 2,
        #    #                    regs= [[-1, n_subset-1]] * (n_rows * max_modify * 2),
        #    #                    tys= [False] * (n_rows * max_modify * 2))
        #    opt_dim = Dimension(size= max_modify * 2,
        #                        regs= [[-1, n_subset-1]] * (max_modify * 2),
        #                        tys= [False] * (max_modify * 2))

    
        #    ''' def opt obj & soluttion to pgb matrix '''
        #    def objective(I):
        #        I_pos, I_neg = optvec2matrix(I, 1, n_cols,
        #                                  return_matrix=False, max_modify= max_modify, subset_idx= subset_idx)
        #        y_pos_ = y_pos ^ cp.asarray(I_pos)
        #        y_neg_ = y_neg ^ cp.asarray(I_neg)

        #        if len(kb_weight)<t+1:
        #            w_trn, w_gem = 1,1
        #        else:
        #            w_trn, w_gem = kb_weight[t][0], kb_weight[t][1]

        #        return w_trn * TRN_reasoner.violated(y_pos_, y_neg_, x_u, I_pos, I_neg) if w_trn!=0 else 0\
        #                + w_gem * GEM_reasoner.violated(y_pos_, y_neg_, z) if w_gem!=0 else 0\
        #                 #+ I_pos.reduce_int()*10 + I_neg.reduce_int()*10

        #    opt_obj = Objective(objective, opt_dim)
        #    if seed:
        #        solution = Opt.min(opt_obj, Parameter(budget=budget, seed=seed))
        #    else:
        #        solution = Opt.min(opt_obj, Parameter(budget=budget))
        #    I_pos, I_neg = optvec2matrix(solution, 1, n_cols,
        #                              return_matrix=False, max_modify= max_modify, subset_idx= subset_idx)

        #    for i in np.nonzero(I_pos)[0]:
        #        mask[idx,i] = 1
        #    for i in np.nonzero(I_neg)[0]:
        #        mask[idx,i] = 1
        #    modified_labels[idx] = modified_labels[idx].union(set(np.nonzero(I_pos^I_neg)[0].tolist()))



        #    ####################

        #    ' modify pseudo label '
        #    y_pos_ = y_pos ^ cp.asarray(I_pos)
        #    y_neg_ = y_neg ^ cp.asarray(I_neg)
        #    y_modified = torch.tensor(y_pos_.astype(int) - y_neg_.astype(int))
        #    Y_modified.append(y_modified)

        #    ##NOTE tmp
        #    #if z != 0:
        #    #    print(f'pos modified: {I_pos.to_lists()[0]}')
        #    #    print(f'neg modified: {I_neg.to_lists()[0]}')

        #' concate Y_modified as tensor, add empty control '
        #Y_modified.append(torch.zeros_like(Y_modified[0]))
        #Y_modified.append(torch.zeros_like(Y_modified[0]))
        #Y_modified.append(torch.zeros_like(Y_modified[0]))
        #Y_modified = torch.stack(Y_modified)
        #X_unlabel_ = torch.concat([X_unlabel, torch.zeros_like(X_unlabel[0:3])], dim=0)

        #NOTE tmp
        #np.save('data_anal/Y_modified_fba.npy', Y_modified.numpy())

        #print(Y_modified[:,test_label_idx])

        #print(Y_pseudo)
        #print(Y_modified)
        #print(f'before modify pos: {Y_pos.reduce_int()}, neg: {Y_neg.reduce_int()}\nafter modify pos: {Y_pos_.reduce_int()} neg: {Y_neg_.reduce_int()}')
        #print(f'non zero before: {np.count_nonzero(Y_pseudo.numpy())}, after: {np.count_nonzero(Y_modified.numpy())}')

        ' retrain base learner '
        learner.load_data(X_unlabel, Y_modified, X_test, Y_test)
        learner.train(KB = reasoner, epochs=retrain_epc, lr=retrain_lr)
        f1 = learner.eval()
        print(f'ABL loop {t}: macro f1 {f1:.4f}')


        #print(learner.predict_prob(X_test)[:,test_label_idx,:])

####################

if __name__ == "__main__":
    log_file = f'log/EGOAL-{datetime.now()}.txt'.replace(' ','-')

    X_train = torch.tensor(np.load('dataset/precise1k/X_label.npy'), dtype = torch.float32)
    Y_train = torch.tensor(np.load('dataset/precise1k/Y_label.npy'), dtype = int)

    X_test = torch.tensor(np.load('dataset/ncbi-sra/X_label.npy'), dtype = torch.float32)
    Y_test = torch.tensor(np.load('dataset/ncbi-sra/Y_label.npy'), dtype = int)

    X_unlabel = torch.tensor(np.load('dataset/X_regulators.npy'), dtype = torch.float32)

    label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
    Y_train = Y_train[:,list(label_set['precise1k_idx'])]
    Y_test = Y_test[:,list(label_set['matrix_idx'])]
    #z_groundtruth = torch.tensor(pd.read_csv('dataset/X_semisup.csv')['growth'], dtype=torch.float32)

    #print(f'Y_train: {Y_train.shape}, Y_test: {Y_test.shape}')

    ##X_unlabel = X_unlabel[2860:2880] # yicR
    ##X_unlabel = X_unlabel[2570:2580]

    #X_unlabel = torch.zeros(size=(10, X_train.shape[1]), dtype=torch.float32)
    #for i in range(10):
    #    X_unlabel[i, 3370+i] = 1 #arcZ
    #    #X_unlabel[i, 2830+i] = 1 #micA
    #    #X_unlabel[i,2960+i] = 1 # gcvB

    #print(list(torch.nonzero(X_unlabel)))


    #z_groundtruth = torch.tensor([0,0,0,-1,0,0,0,0,0,0]) #arcZ
    ##z_groundtruth = [0,0,0,0,0,0,0,-1,0,0] #micA

    ##idx_list = [26,27] # yicR
    ##idx_list = [51,52,53,54,55,56] #gcvB
    ##idx_list = [0,1,2,4,5,11,12] # mazF

    #idx_list = list(range(42,54)) # arcZ
    ##idx_list = list(range(60,63)) # micA

    #print(torch.nonzero(X_test[idx_list]).tolist())
    #print(torch.nonzero(X_test[[i for i in range(len(X_test)) if i not in idx_list]]).tolist())
    #X_test = X_test[idx_list]
    #Y_test = Y_test[idx_list]

    #kb_weights = [(1,0), (1,0), (.5,.5), (0,1), (0,1)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    X_unlabel = X_unlabel.to(device)

    abduce(X_unlabel= X_unlabel,
           X_test= X_test,
           Y_test= Y_test,
           pos_trn_pth='rules/regu_pos.npz',
           neg_trn_pth='rules/regu_neg.npz',
           X_label = X_train,
           Y_label = Y_train,
           T=5,
           max_modify=20,
           budget=1000,
           pretrain_epc=2000,
           pretrain_lr=1e-3,
           use_gpu=True,
           #subset_threshold=[1.,.9,.9],
           subset_threshold = 1.,
           retrain_epc=500,
           retrain_lr=1e-3,
           seed=42, log_file=log_file)
