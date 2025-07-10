import numpy as np
import torch
from scipy.sparse import save_npz, load_npz
import cupy as cp
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
import pandas as pd
import time


class RegualtoryKB():
    def __init__(self,
                 pos_trn_pth: str,
                 neg_trn_pth: str,
                 T=None,
                 use_gpu=False) -> None:
        '''
        Class for Regulatory Network Knowledgebase
        Args:
            pos_trn_pth:
            neg_trn_pth:
            T:
            use_gpu:
        '''

        self.Regu_P_0 = torch.tensor(load_npz(pos_trn_pth).toarray(), dtype=torch.float)
        self.Regu_N_0 = torch.tensor(load_npz(neg_trn_pth).toarray(), dtype=torch.float)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_gpu:
            self.Regu_P_0 = self.Regu_P_0.to(self.device)
            self.Regu_N_0 = self.Regu_N_0.to(self.device) 

        #' transitive closure '
        #self.KB_P, self.KB_N, self.T = self.closure(Regu_P, Regu_N, T=T)

        ' filter output genes from cols (align with Y) '
        label_set = pd.read_csv('dataset/label_set_iml.csv')
        self.idx_list = list(label_set['matrix_idx'])

        ' initialize pos & neg KB '
        self.KB_P = self.Regu_P_0[:,self.idx_list]
        self.KB_N = self.Regu_N_0[:,self.idx_list]
        self.KB = torch.clamp(self.KB_P - self.KB_N, -1.,1.)


    def closure_(self, T=None):
        ''' Inplace & Nonstatic Ver of KB Closure '''
        R_P, R_N ,_ = self.closure(self.Regu_P_0, self.Regu_N_0, T=5, device=self.device)
        R_P_2, R_N_2,_ = self.closure(self.Regu_P_0, self.Regu_N_0, T=2, device=self.device)
        self.KB_P, self.KB_N = R_P, R_N

        R_diff = R_P - R_N
        R_P, R_N = R_P.bool(), R_N.bool()
        self.KB = torch.where(R_P & R_N,
                     torch.where(R_P_2.bool()&R_N_2.bool(), 
                                 torch.clamp(self.Regu_P_0-self.Regu_N_0,-1,1),
                                 torch.clamp(R_P_2-R_N_2,-1,1)), R_diff)

        self.KB_P, self.KB_N, self.T = self.closure(self.Regu_P_0, self.Regu_N_0, T, self.device)
        self.KB_P, self.KB_N  = self.KB_P[:,self.idx_list], self.KB_N[:,self.idx_list]

        self.KB = torch.clamp(self.KB_P - self.KB_N, -1.,1.)
        return self.KB_P, self.KB_N, self.T


    def violated(self,
                 Y: torch.Tensor,
                 X: torch.Tensor,
                 mask=None | torch.Tensor):
        '''
        violated count across all data points (matrix)
        Args:
            Y: 
            X:
            mask:
        '''

        #if Y.shape != X.shape or (mask!=None and (mask.shape != Y.shape\
        #        and (mask.shape[1]!=1 or mask.shape[0]!=Y.shape[0]))):
        #    raise(Exception('All matrices should be in same shape'))

        deduction = torch.clamp(X @ self.KB, -1.,1.)
        if mask != None:
            vio_cnt = torch.count_nonzero(deduction != Y)
        else:
            vio_cnt = torch.count_nonzero((deduction != Y)[mask])
        return vio_cnt
        #vio_mat_pos = (~ Y_pos) & (X @ self.KB_P & ~(X @ self.KB_N))
        #vio_mat_neg = (~ Y_neg) & (X @ self.KB_N & ~(X @ self.KB_P))
        #vio_mat_zero = (Y_pos | Y_neg) & ~(X @ self.KB_P) & ~(X @ self.KB_N)
        #vio_mat_dual = (I_pos | I_neg) & (X @ self.T_P & X @ self.T_N)
        #vio_cnt = cp.count_nonzero(vio_mat_pos)\
        #        + cp.count_nonzero(vio_mat_neg)\
        #        + cp.count_nonzero(vio_mat_zero)\
                #+ cp.count_nonzero(vio_mat_dual)
        #return int(vio_cnt)


    def deduce(self, X: torch.Tensor):
        ''' deduction result (multiplication) '''
        return X @ self.KB


    @staticmethod
    def closure(R_P_0: torch.Tensor,
                R_N_0: torch.Tensor,
                T=None,
                device=torch.device('cpu')):
        '''
        Transitive Closure of Regulatory Matrix

        Args:
            R_P_0:
            R_N_0:
            T:
            device:

        Return Values:
            R_P:
            R_N:
            cnt:
        '''

        cnt = 1
        R_P_0, R_N_0 = R_P_0.to(device), R_N_0.to(device)
        I = torch.eye(R_P_0.shape[0]).to(device)
        
        R_P, R_N = R_P_0+I, R_N_0
        while True:
            if T!=None and cnt >= T:
                break
    
            R_P_, R_N_ = R_P, R_N
        
            R_P = R_P_0 @ R_P_ + R_N_0 @ R_N_
            R_N = R_P_0 @ R_N_ + R_N_0 @ R_P_
        
            R_P, R_N = torch.clamp(R_P,0,1), torch.clamp(R_N,0,1)
            #R_P /= torch.min(R_P[R_P!=0])
            #R_N /= torch.min(R_N[R_N!=0])
        
            if torch.all(R_P == R_P_) and torch.all(R_N == R_N_):
                break
            cnt += 1
        
        R_N = torch.where(I.bool(), 0, R_N)
        return R_P, R_N, cnt



#class MetabolicKB():
#    def __init__(self, pos_gem_pth, neg_gem_pth, annotation_pth, T=None) -> None:
#        GEM_P =  load_npz(pos_gem_pth).toarray()
#        GEM_N =  load_npz(neg_gem_pth).toarray()
#        self.A = cp.asarray(load_npz(annotation_pth).toarray())
#
#        ''' transitive closure '''
#        self.KB_P, self.KB_N, self.T = self.closure(GEM_P, GEM_N, T=T)
#        self.KB_P, self.KB_N = self.A @ self.KB_P, self.A @ self.KB_N
#        self.KB_P, self.KB_N = self.KB_P.astype(cp.int32), self.KB_N.astype(cp.int32)
#
#        # NOTE tmp
#        neg = np.nonzero(self.KB_N[:,1])[0].tolist()
#        pos = np.nonzero(self.KB_P[:,1])[0].tolist()
#        print('closure result:',len(set(pos)-set(neg)), len(set(neg)-set(pos)))
#        # NOTE end
#
#
#    def closure(self, R0_P, R0_N, T=None, print_matrix=False):
#
#        ''' get dim of matrix KB
#            note. positive & negative regultion KB should of same size '''
#        dim = R0_P.shape[0]
#        assert dim==R0_N.shape[0]
#    
#        R0_P = cp.asarray(R0_P, dtype=cp.bool_)
#        R0_N = cp.asarray(R0_N, dtype=cp.bool_)
#    
#        ''' R*_P = sum ((R_P)^i R_P + (R_N)^i R_N)
#            R*_N = sum ((R_P)^i R_N + (R_N)^i R_P) '''
#        I = cp.eye(dim, dtype=cp.bool_)
#        R_P = I + R0_P
#    
#        if print_matrix:
#            print(f'initial R_P = R_P + I = \n{str(R_P)}\ninitial R_N =\n{str(R_N)}\n----------\n')
#    
#        cnt = 0
#        while True:
#            ' multiply until closure '
#            R_P_ = R_P @ (R0_P+I) #+ R_N @ (R0_N+I)
#            #R_N_ = R_P @ (R0_N+I) + R_N @ (R0_P+I)
#            if print_matrix:
#                print(f'{cnt}th iteration:\nR{cnt}_P =\n{str(R_P_)}\nR{cnt}_N =\n\n')
#            if cp.all(R_P == R_P_):# and R_N_.iseq(R_N):
#                break
#            if T and cnt>=T:
#                break
#            cnt += 1
#            R_P = R_P_
#            #R_N = R_N_
#    
#        res_P = R_P 
#        res_N = (R0_N @ R_P) & ~I
#        
#        # TODO
#        #intersection = res_P & res_N
#        #res_P = res_P & negation(intersection)
#        #res_N = res_N & negation(intersection)
#        #res_N = R0_N & negation(I)
#
#        if print_matrix:
#            print(f'\n----------\nfinal result:\nR*_P =\n{str(res_P)}\nR*_N =\n{str(res_N)}\n')
#        return res_P, res_N, cnt
#
#    def violated(self, Y_pos, Y_neg, z):
#        ''' violated count across all data points (matrix) '''
#        pos = (Y_pos @ self.KB_P)[0]\
#                     #+ (Y_neg @ self.KB_N)[0]
#        neg = (Y_neg @ self.KB_P)[0]\
#                     #+ (Y_pos @ self.KB_N)[0]
#
#        pos,neg = int(pos),int(neg)
#        z_pseudo = 1 if pos>neg else (-1 if pos<neg else 0)
#
#        return 0 if z_pseudo==z else abs(pos-neg)*10
#
#    def deduce(self, Y_pos, Y_neg):
#        ''' deduction result (multiplication) '''
#        pos = (Y_pos @ self.KB_P)[0]\
#                     + (Y_neg @ self.KB_N)[0]
#        neg = (Y_neg @ self.KB_P)[0]\
#                     + (Y_pos @ self.KB_N)[0]
#
#        return pos, neg

if __name__ == '__main__':
    cp.cuda.Device(0).use()
    
    regulatoryKB = RegualtoryKB(pos_trn_pth='rules/regu_pos.npz',
                                neg_trn_pth='rules/regu_neg.npz')

    print(regulatoryKB.KB_P, regulatoryKB.KB_P.shape)
    print(len(cp.nonzero(regulatoryKB.KB_P)[0]))
    print('closure times:', regulatoryKB.T)

    print(np.count_nonzero(np.sum(regulatoryKB.KB_P & regulatoryKB.KB_N, axis=0)))
    print(np.count_nonzero(np.sum(regulatoryKB.KB_P | regulatoryKB.KB_N, axis=0)))
    #save_npz('rules/regu_pos_clo.npz', cp_coo_matrix(regulatoryKB.KB_P))
    #save_npz('rules/regu_neg_clo.npz', cp_coo_matrix(regulatoryKB.KB_N))

    #metabolicKB = MetabolicKB(pos_gem_pth='rules/gem_pos.npz', neg_gem_pth='rules/gem_neg.npz', annotation_pth='rules/gem_annot.npz')
    #print(metabolicKB.KB_P, metabolicKB.KB_P.shape)
    #print(len(cp.nonzero(metabolicKB.KB_P)[0]))
    #print(type(metabolicKB.KB_P), metabolicKB.KB_P.device)
    #print('closure times:', metabolicKB.T)

    exit()
    import random
    import time
    pos_lst = sorted(random.sample(range(622), k=random.randint(1, 622)))
    pos_vec = pgb.Vector.from_lists(pos_lst, True, size=623, typ=pgb.BOOL)
    neg_lst = sorted(random.sample(range(622), k=random.randint(1, 622)))
    neg_vec = pgb.Vector.from_lists(neg_lst, True, size=623, typ=pgb.BOOL)

    t_0 = time.time()
    res = metabolicKB.deduce(pos_vec,neg_vec)
    t = time.time() - t_0

    print(res)
    print(f'time: {t}')
    



    #''' demos for KB closure '''

    #' FFL: A + B, B + C, A - C'
    #R0_P = pgb.Matrix.from_lists([0,1],[1,2], ncols=3, nrows=3)
    #R0_N = pgb.Matrix.from_lists([0],[2], ncols=3, nrows=3)
    #print(str(R0_P), str(R0_P.lnot()), negation(R0_P))

    #R_P,R_N = closure(R0_P,R0_N, print_matrix=False)

    #' FFL: A + B, B - C, A + C'
    #R0_P = pgb.Matrix.from_lists([0,0],[1,2], ncols=3, nrows=3)
    #R0_N = pgb.Matrix.from_lists([1],[2], ncols=3, nrows=3)
    #R_P,R_N = closure(R0_P,R0_N, print_matrix=False)

    #' neg FBL '
    #R0_P = pgb.Matrix.from_lists([0,1],[1,2], ncols=3, nrows=3)
    #R0_N = pgb.Matrix.from_lists([2],[0], ncols=3, nrows=3)
    #R_P,R_N = closure(R0_P,R0_N, print_matrix=False)

    #' NAR '
    #R0_P = pgb.Matrix.from_lists([0,1],[1,2], ncols=3, nrows=3)
    #R0_N = pgb.Matrix.from_lists([1],[1], ncols=3, nrows=3)
    #R_P,R_N = closure(R0_P,R0_N, print_matrix=True)

    #''' demos for FBA KB '''
    #fluxKB = FluxKB(bound_path='rules/fba_bound.json')

    #import random
    ##NOTE tmp for sampling
    #pos_lst = sorted(random.sample(range(240), k=random.randint(1, 240)))
    #pos_vec = pgb.Vector.from_lists(pos_lst, True, size=241, typ=pgb.BOOL)
    #neg_lst = sorted(random.sample(range(240), k=random.randint(1, 240)))
    #neg_vec = pgb.Vector.from_lists(neg_lst, True, size=241, typ=pgb.BOOL)

    #print(fluxKB.violated(pos_vec, neg_vec))

