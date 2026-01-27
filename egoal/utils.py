from typing import Union
#import pygraphblas as pgb
import torch
import cupy as cp
from cupyx.scipy.sparse import coo_matrix
import numpy as np
import json

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
