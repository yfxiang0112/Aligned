from typing import Union
#import pygraphblas as pgb
import torch
import cupy as cp
from cupyx.scipy.sparse import coo_matrix
import numpy as np
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from egoal.reasoner import RegualtoryKB

def eval_log(Y_true,
             Y_pred,
             log_file,
             Y_prob = None):

    total = Y_pred.size(0)
    correct = (Y_true == Y_pred).sum(dim=0)

    ''' compute total confusion matrix '''
    flat_y_t = Y_true.flatten()
    flat_y_p = Y_pred.flatten()
    confusion = confusion_matrix(flat_y_t, flat_y_p, labels=[-1, 0,1])
    confusion = confusion / confusion.sum().sum()
    f1_macro = f1_score(flat_y_t, flat_y_p, average='macro') # micro on labels, macro on classes
    f1_micro = f1_score(flat_y_t, flat_y_p, average='micro') # micro on labels, micro on classes

    ' compute weighted f1 by ground truth proportion '
    weights = [1/(torch.sum(flat_y_t==-1).item() + 1e-6),
               1/(torch.sum(flat_y_t==0).item() + 1e-6),
               1/(torch.sum(flat_y_t==1).item() + 1e-6)]
    weights = torch.Tensor(weights) / sum(weights)
    f1_class = f1_score(flat_y_t, flat_y_p, average=None)
    f1_weighted = sum([f1*w for f1,w in zip(f1_class,weights)])
    #f1_weighted = f1_class[0]*weights[0] + f1_class[2]*weights[2]


    ''' compute acc & confusion matrix on each gene '''
    per_label_accuracy = correct / total

    with open(log_file,'a') as f:
        f.write('label ')
        for i in range(len(per_label_accuracy)):
            f.write(f'{i:8}\t')
        f.write('\n   acc ')
        for acc in per_label_accuracy:
            f.write(f'{acc * 100:7.2f}%\t')
        f.write('\n    f1 ')
        for label_idx in range(Y_true.shape[1]):
            f.write(f"{f1_score(Y_true[:,label_idx], Y_pred[:,label_idx], average='macro'):8.4f}\t")
        #f.write('\n---- data ----')

        for data_idx in range(Y_true.shape[0]):
            f.write(f'\npred{data_idx:2} ')
            for y_pred in Y_pred[data_idx]:
                f.write(f'{y_pred:8}\t')
            if Y_prob != None:
                f.write(f'\nprob{data_idx:2} ')
                for y_prob in Y_prob[data_idx]:
                    f.write(f'{y_prob:8.2f}\t')
            f.write(f'\ntest{data_idx:2} ')
            for y_test in Y_true[data_idx]:
                f.write(f'{y_test:8}\t')

        f.write(f'\n------\nconfusion matrix:\n{confusion}\n')
        f.write(f'macro f1: {f1_macro}\n')
        f.write(f'micro f1: {f1_micro}\n')
        f.write(f'weighted f1: {f1_weighted}\n')
        f.write(f'class -1 f1: {f1_class[0]}\n')
        f.write(f'class  0 f1: {f1_class[1]}\n')
        f.write(f'class  1 f1: {f1_class[2]}\n')
        f.write(f'average label-wise acc: {np.mean(np.array(per_label_accuracy))*100:.2f}%\n')

    return f1_macro

def label_weight(
        X_test : torch.Tensor,
        Y_test : torch.Tensor,
        KB: RegualtoryKB,
        X_train = None | torch.Tensor,
        Y_train = None | torch.Tensor,
        GOA_path = None | str):

    if X_train != None:
        total = len(Y_train)
        Y_deduction = KB.deduce(X_train)
        kb_train = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_train), axis=0) / total > .01)[0].tolist())
    else:
        kb_train = list(range(Y_test.shape[1]))
                        
    total = len(Y_test)
    Y_deduction = KB.deduce(X_test)
    kb_test = (np.nonzero(np.sum((Y_deduction != 0) & (Y_deduction == Y_test), axis=0) / total > .1)[0].tolist())
    kb_con_idx = list(set(kb_train).intersection(kb_test))


    ' weight with GO annotation '
    gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
    go_annot = pd.read_csv('rules/GO/goa_gene2go.csv', index_col = 0)
    #go_annot = go_annot.loc[gene_idx.loc[gene_idx['iml1515_idx']!=-1, 'locus']]
    go_annot_num = np.array([len(eval(go_annot.loc[i,'concepts'])) if i in go_annot.index else 0 for i in gene_idx.loc[gene_idx['iml1515_idx']!=-1, 'locus']], dtype=np.float32)
    go_annot_num /= np.max(go_annot_num)
    #go_annot_num = np.max(go_annot_num - .3, np.zeros_like(go_annot_num))
    print(go_annot_num)
    
    
    ' weight with in-degree in GRN '
    iml_idx = list(gene_idx[gene_idx['iml1515_idx']!=-1].index)
    regulatory_p = KB.Regu_P_0.cpu().numpy()
    regulatory_n = KB.Regu_N_0.cpu().numpy()
    regulatory_num = np.sum(regulatory_p + regulatory_n, axis=0)[iml_idx]
    regulatory_num /= np.max(regulatory_num)
    #regulatory_num = np.max(regulatory_num - .3, np.zeros_like(regulatory_num))
    print(regulatory_num)

    weights = np.full(shape=Y_true.shape[1], fill_value=-.3, dtype=np.float32)
    weights += (go_annot_num - .3) + (regulatory_num - .3)
    weights[kb_con_idx] += 1.3
    weights = np.clip(weights, -1., 1.)

    return weights
    return np.zeros(shape=Y_test.shape[1])

#def negation(M: Union[pgb.Matrix, pgb.Vector], print_matrix=False) -> Union[pgb.Matrix, pgb.Vector]:
#    if type(M) == pgb.Matrix:
#        N = pgb.Matrix.dense(pgb.BOOL, nrows=M.nrows, ncols=M.ncols, fill=True)
#        idx = M.to_lists()
#        if print_matrix:
#            print(idx[0])
#            print(idx[1])
#            print(idx[2])
#        for i,j in zip(idx[0], idx[1]):
#            del N[i,j]
#        return N
#    else:
#        N = pgb.Vector.dense(pgb.BOOL, size=M.size, fill=True)
#        idx = M.to_lists()[0]
#        for i in idx:
#            del N[i]
#        return N
#
#
#def identity(dim):
#    I = pgb.Matrix.sparse(pgb.BOOL, dim, dim)
#    for i in range(dim):
#        I[i, i] = True
#    return I
#
#def cp_identity(dim):
#    rows = cp.arange(dim, dtype=cp.int32)
#    cols = cp.arange(dim, dtype=cp.int32)
#    data = cp.ones(dim, dtype=cp.float32)
#    return coo_matrix((data, (rows, cols)), shape=(dim, dim))
#
#    
#        
#def xor(M1: Union[pgb.Matrix, pgb.Vector], M2: Union[pgb.Matrix, pgb.Vector]) -> Union[pgb.Matrix, pgb.Vector]:
#    ''' M2 as the sparser matrix / vector '''
#    assert type(M1) == type(M2)
#
#    if type(M1) == pgb.Matrix:
#        assert M1.shape == M2.shape
#        idx_1 = M1.to_lists()
#        idx_1= set(zip(idx_1[0], idx_1[1]))
#        idx_2 = M2.to_lists()
#
#        for i,j in zip(idx_2[0], idx_2[1]):
#            if (i,j) in idx_1:
#                idx_1.remove((i,j))
#            else:
#                idx_1.add((i,j))
#
#        if len(idx_1) > 0:
#            row, col = zip(*list(idx_1))
#            return pgb.Matrix.from_lists(row,col, nrows=M1.shape[0], ncols=M1.shape[1], typ=pgb.BOOL)
#        else:
#            return pgb.Matrix.sparse(typ=pgb.BOOL, nrows=M1.shape[0], ncols=M1.shape[1])
#    else:
#        assert M1.size == M2.size
#        idx_1 = set(M1.to_lists()[0])
#        idx_2 = M2.to_lists()[0]
#        for i in idx_2:
#            if i in idx_1:
#                idx_1.remove(i)
#            else:
#                idx_1.add(i)
#        if len(idx_1) > 0:
#            return pgb.Vector.from_lists(list(idx_1), True, size=M1.size, typ=pgb.BOOL)
#        else:
#            return pgb.Vector.sparse(typ=pgb.BOOL, size=M1.size)
#
#
#########################################
#
#
#def matrix2pgb(M_in: Union[np.ndarray, torch.Tensor]) -> pgb.Matrix:
#    if type(M_in) == torch.Tensor:
#        M_in = M_in.numpy()
#    M_in = M_in.astype(int)
#    if len(M_in.shape) == 2:
#        rows, cols = np.nonzero(M_in)
#        if len(rows) == 0:
#            return pgb.Matrix.sparse(nrows=M_in.shape[0], ncols=M_in.shape[1], typ=pgb.BOOL)
#        return pgb.Matrix.from_lists(rows.tolist(), cols.tolist(), nrows=M_in.shape[0], ncols=M_in.shape[1], typ=pgb.BOOL)
#    elif len(M_in.shape) == 1:
#        idx, = np.nonzero(M_in)
#        if len(idx) == 0:
#            return pgb.Vector.sparse(size=M_in.shape[0], typ=pgb.BOOL)
#        return pgb.Vector.from_lists(idx.tolist(), True, size=M_in.shape[0], typ=pgb.BOOL)

#def pgb2torch(M_pgb: pgb.Matrix) -> torch.Tensor:
#    M = M_pgb.to_numpy()
#    return torch.tensor(M)
    #''' Extract row indices, column indices, and values '''
    #rows, cols, values = M_pgb.to_values()

    #''' Convert to PyTorch sparse tensor format '''
    #indices = torch.tensor([rows, cols], dtype=torch.long)  # Shape: (2, num_nonzero)
    #values = torch.tensor(values, dtype=torch.float32)  # Shape: (num_nonzero,)
    #size = (M_pgb.nrows, M_pgb.ncols)

    #''' Create and return a sparse PyTorch tensor '''
    #return torch.sparse_coo_tensor(indices, values, size)

#def torch2pgb(M_torch: torch.Tensor) -> pgb.Matrix:
#    ''' Extract indices and values from the sparse PyTorch tensor '''
#    print(M_torch.indices())
#    indices = M_torch.indices().tolist()
#    values = M_torch.values().tolist()
#
#    ''' Create and return a PyGraphBLAS sparse matrix '''
#    return pgb.Matrix.from_values(indices[0], indices[1], values)

#def pgb2ternary(M_pos: Union[pgb.Matrix, pgb.Vector], M_neg: Union[pgb.Matrix, pgb.Vector]) -> torch.Tensor:
#    assert type(M_pos) == type(M_neg)
#    if type(M_pos) == pgb.Matrix:
#        np_pos = M_pos.to_numpy()
#        np_neg = M_neg.to_numpy()
#        res = np.zeros_like(np_pos, dtype=int)
#        #with open('dataset/Y_max.json','r') as f:
#        #    dual_regu = json.load(f)
#        #for i in range(res.shape[1]):
#        #    res[:,i] = np.where((np_pos[:,i]==1) & (np_neg[:,i]==1), dual_regu[str(i)],\
#        #                        np.where(np_pos[:,i]==1, 1,\
#        #                        np.where(np_neg[:,i]==1, -1, 0)))
#        res[:,:] = np.where((np_pos[:,:]==1) & (np_neg[:,:]==1), 0,\
#                            np.where(np_pos[:,:]==1, 1,\
#                            np.where(np_neg[:,:]==1, -1, 0)))
#
#        return torch.tensor(res, dtype=int)
#    else:
#        res = torch.zeros(M_pos.size, dtype=int)
#        set_p = set(M_pos.to_lists()[0])
#        set_n = set(M_neg.to_lists()[0])
#        idx_p = set_p - set_n
#        idx_n = set_n - set_p
#        idx_d = set_p.intersection(set_n)
#        #res[idx_p] = -1
#        #res[idx_n] = 1
#
#        res[list(idx_p)] = 1
#        res[list(idx_n)] = -1
#        res[list(idx_d)] = 0
#        #with open('dataset/Y_max.json','r') as f:
#        #    dual_regu = json.load(f)
#        #    for i in idx_d:
#        #        #print(i, dual_regu[str(i)])
#        #        res[i] = dual_regu[str(i)]
#        return res
#
#def optvec2pgb(x, n_rows, n_cols, return_matrix=True, max_modify=1, subset_idx=[]):
#    ''' optimization vector to I_pos and I_neg '''
#    x_values = x.get_x()
#    x_pos = x_values[:n_rows*max_modify]
#    x_neg = x_values[n_rows*max_modify:]
#
#    if subset_idx == []:
#        subset_idx = range(n_cols)
#
#    def sol2pgb(x):
#        if return_matrix:
#            rows = [ i // max_modify   for i in range(n_rows * max_modify) if x[i]!=-1]
#            cols = [ subset_idx[x[i]]  for i in range(n_rows * max_modify) if x[i]!=-1]
#            if len(rows) == 0:
#                return pgb.Matrix.sparse(nrows=n_rows, ncols=n_cols, typ=pgb.BOOL)
#            return pgb.Matrix.from_lists(rows, cols, nrows=n_rows, ncols=n_cols, typ=pgb.BOOL)
#        else:
#            idx = [subset_idx[x[i]] for i in range(max_modify) if x[i] != -1]
#            if len(idx) == 0:
#                return pgb.Vector.sparse(size=n_cols, typ=pgb.BOOL)
#            return pgb.Vector.from_lists(idx, True, size=n_cols, typ=pgb.BOOL)
#
#    return sol2pgb(x_pos), sol2pgb(x_neg)

def optvec2matrix(x, n_rows, n_cols, return_matrix=True, max_modify=1, subset_idx=[]):
    ''' optimization vector to I_pos and I_neg '''
    x_values = x.get_x()
    x_pos = x_values[:n_rows*max_modify]
    x_neg = x_values[n_rows*max_modify:]

    if subset_idx == []:
        subset_idx = range(n_cols)

    def sol2pgb(x):
        if return_matrix:
            rows = [ i // max_modify   for i in range(n_rows * max_modify) if x[i]!=-1]
            cols = [ subset_idx[x[i]]  for i in range(n_rows * max_modify) if x[i]!=-1]
            shape = (n_rows, n_cols)
            res = np.zeros(shape, dtype=np.bool_)
            if len(rows) != 0:
                res[rows,cols] = True
        else:
            idx = [subset_idx[x[i]] for i in range(max_modify) if x[i] != -1]
            res = np.zeros(n_cols, dtype=np.bool_)
            if len(idx) != 0:
                res[idx] = True
        return res

    return sol2pgb(x_pos), sol2pgb(x_neg)
    

#if __name__ == '__main__':
#    vec1 = pgb.Vector.from_lists([0,2,3,5,7,8,10,13], True, size=15, typ=pgb.BOOL)
#    vec2 = pgb.Vector.from_lists([0,1,5,9], True, size=15, typ=pgb.BOOL)
#    print(xor(vec1, vec2))
