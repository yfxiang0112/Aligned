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
           output_idx_list = None,
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
                            output_idx_list= output_idx_list,
                            use_gpu=use_gpu)#, T=4)
    reasoner.closure_(T=5, closure_type='weighted')

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

        #R_mask = torch.zeros_like(R_binary, dtype=torch.bool).to(device)
        #labels = [3, 13, 57, 60, 70, 93, 114, 119, 142, 160, 162, 164, 173, 175, 179, 193, 197, 198, 209, 213, 218, 228, 262, 263, 265, 274, 284, 285, 289, 291, 315, 320, 339, 341, 344, 353, 361, 369, 370, 374, 385, 405, 414, 425, 430, 436, 437, 439, 443, 445, 446, 467, 481, 482, 485, 503, 504, 507, 514, 518, 521, 530, 531, 540, 541, 549, 551, 552, 555, 561, 570, 578, 579, 583, 596, 601, 608, 641, 643, 645, 652, 662, 663, 668, 670, 672, 673, 674, 685, 689, 690, 702, 703, 705, 706, 708, 712, 723, 724, 727, 732, 733, 734, 737, 738, 747, 748, 751, 752, 782, 801, 807, 808, 823, 826, 854, 873, 882, 884, 905, 909, 912, 915, 917, 919, 921, 945, 953, 965, 968, 975, 983, 997, 998, 999, 1013, 1029, 1030, 1031, 1039, 1040, 1046, 1055, 1064, 1066, 1077, 1119, 1133, 1148, 1162, 1173, 1174, 1177, 1178, 1180, 1181, 1182, 1206, 1212, 1219, 1220, 1221, 1224, 1229, 1233, 1237, 1239, 1251, 1275, 1276, 1277, 1278, 1282, 1315, 1342, 1347, 1351, 1352, 1353, 1362, 1363, 1364, 1367, 1380, 1400, 1403, 1407, 1426, 1431, 1432, 1443, 1444, 1445, 1466, 1478, 1487, 1506]
        #R_mask[:,labels] = True

        #print(R)
        #print(R_binary)
        print(torch.count_nonzero(R_binary) / (R.shape[0]*R.shape[1]))
        #exit()
        n_rows, n_cols = Y_pseudo.shape[0], Y_pseudo.shape[1]

        Y_deduction = reasoner.deduce(X_unlabel)

        Y_modified = torch.where(R_binary , Y_deduction, Y_pseudo)
        #Y_modified = Y_deduction

        print(torch.count_nonzero(Y_deduction))
        #exit()
        np.save(f'data_anal/abduction_results/R_ABL{t}.npy', R_binary.cpu().numpy())
        np.save(f'data_anal/abduction_results/Yp_ABL{t}.npy', Y_pseudo.cpu().numpy())
        np.save(f'data_anal/abduction_results/Yd_ABL{t}.npy', Y_deduction.cpu().numpy())

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

    test_idx = [37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]
    # arcZ, gcvB, micA, ryhB
    X_test, Y_test = X_test[test_idx], Y_test[test_idx]

    X_unlabel = torch.tensor(np.load('dataset/X_regulators.npy'), dtype = torch.float32)

    #NOTE tmp
    X_unlabel = torch.zeros(size=(len(test_idx), X_train.shape[1]), dtype=torch.float32)
    for i in range(0,12):
        X_unlabel[i,3373] = 1.
    for i in range(12,18):
        X_unlabel[i,2961] = 1.
    for i in range(18,21):
        X_unlabel[i,2837] = 1.
    for i in range(21,28):
        X_unlabel[i,2606] = 1.
    #print(X_unlabel.shape)
    #print(torch.nonzero(X_unlabel))

    #X_unlabel = X_train

    label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
    #labels = [1, 2, 23, 43, 48, 50, 82, 83, 84, 97, 107, 134, 135, 136, 137, 143, 149, 159, 160, 161, 162, 163, 165, 166, 180, 181, 182, 184, 185, 186, 191, 192, 222, 223, 224, 225, 226, 227, 238, 243, 246, 255, 258, 259, 263, 264, 265, 266, 267, 268, 269, 270, 272, 275, 284, 286, 287, 301, 302, 303, 318, 328, 331, 337, 338, 342, 343, 348, 353, 357, 359, 360, 362, 363, 386, 387, 389, 402, 403, 407, 411, 417, 418, 423, 424, 432, 436, 443, 444, 448, 449, 452, 457, 458, 464, 471, 472, 473, 478, 479, 480, 488, 501, 542, 549, 550, 558, 560, 561, 562, 564, 570, 572, 573, 579, 584, 585, 586, 587, 588, 589, 593, 594, 595, 600, 605, 606, 621, 622, 624, 627, 630, 632, 633, 634, 641, 648, 650, 657, 659, 661, 683, 694, 695, 696, 697, 698, 700, 705, 709, 710, 719, 720, 728, 734, 739, 741, 748, 752, 753, 755, 762, 763, 764, 765, 766, 767, 775, 776, 799, 809, 811, 813, 818, 832, 833, 850, 851, 862, 863, 864, 865, 870, 892, 893, 894, 895, 903, 904, 911, 925, 927, 934, 936, 945, 946, 950, 951, 952, 953, 954, 955, 963, 965, 977, 978, 986, 987, 989, 990, 991, 992, 996, 997, 998, 999, 1003, 1005, 1006, 1007, 1008, 1011, 1012, 1013, 1014, 1015, 1016, 1029, 1030, 1031, 1032, 1033, 1043, 1047, 1052, 1053, 1058, 1059, 1060, 1061, 1066, 1071, 1079, 1083, 1091, 1094, 1095, 1101, 1102, 1103, 1108, 1111, 1121, 1123, 1132, 1133, 1134, 1136, 1140, 1141, 1142, 1161, 1171, 1172, 1175, 1178, 1183, 1202, 1203, 1204, 1209, 1213, 1215, 1231, 1240, 1245, 1246, 1248, 1250, 1251, 1252, 1262, 1274, 1307, 1316, 1318, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1328, 1329, 1339, 1340, 1342, 1347, 1348, 1349, 1357, 1358, 1359, 1366, 1371, 1372, 1377, 1381, 1382, 1388, 1389, 1391, 1392, 1396, 1402, 1403, 1409, 1410, 1411, 1412, 1429, 1430, 1434, 1437, 1438, 1439, 1442, 1443, 1452, 1461, 1462, 1468, 1474, 1475, 1495, 1497, 1498, 1502, 1507, 1511]
    idx_list_p1k = list(label_set['precise1k_idx'])
    idx_list_sra = list(label_set['matrix_idx'])

    Y_train = Y_train[:,idx_list_p1k]
    Y_test = Y_test[:,idx_list_sra]

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
           T=1,
           max_modify=20,
           budget=1000,
           pretrain_epc=5000,
           pretrain_lr=1e-3,
           output_idx_list=idx_list_sra,
           use_gpu=True,
           #subset_threshold=[1.,.9,.9],
           subset_threshold = 1.,
           retrain_epc=10000,
           retrain_lr=1e-4,
           seed=42, log_file=log_file)
