import numpy as np
import torch
from scipy.sparse import coo_matrix, save_npz, load_npz
import pandas as pd
import time
from sklearn.metrics import f1_score
import networkx as nx

def exp_soft(X,t):
    return 1 - torch.exp(-t * X)

def exp_power(X, k, t):
    Xk = torch.matrix_power(X, k)
    return 1 - torch.exp(-t * Xk)

def exp_closure(X_P, X_N, k, t):
    X_P_0, X_N_0 = X_P, X_N
    for _ in range(k):
    
        X_P_, X_N_ = X_P, X_N
    
        X_P = X_P_0 @ X_P_ + X_N_0 @ X_N_
        X_N = X_P_0 @ X_N_ + X_N_0 @ X_P_
    
        if torch.all(X_P == X_P_) and torch.all(X_N == X_N_):
            break
    
    return 1 - torch.exp(-t * X_P), 1 - torch.exp(-t * X_N)

def tanh_soft(X, t):
    return torch.tanh(t * X)

def tanh_power(X, k, t):
    return torch.tanh(t * torch.matrix_power(X, k))


class RegulatoryKB():
    '''
    Class for Regulatory Network Knowledgebase
    Args:\n
        pos_trn_pth: npz file path 
        neg_trn_pth:
        output_idx_list:
        device:
    '''
    def __init__(self,
                 pos_trn_pth: str,
                 neg_trn_pth: str | None,
                 output_idx_list=None,
                 device='cpu') -> None:

        self.Regu_P_0 = torch.tensor(load_npz(pos_trn_pth).toarray(), dtype=torch.float)
        self.Regu_N_0 = torch.tensor(load_npz(neg_trn_pth).toarray(), dtype=torch.float) if neg_trn_pth!=None else torch.zeros_like(self.Regu_P_0)

        self.device = device
        if self.device != 'cpu':
            self.Regu_P_0 = self.Regu_P_0.to(self.device)
            self.Regu_N_0 = self.Regu_N_0.to(self.device) 

        self.Regu_0 = torch.clamp(torch.abs(self.Regu_P_0)-torch.abs(self.Regu_N_0), -1,1)

        #' transitive closure '
        #self.KB_P, self.KB_N, self.T = self.closure(Regu_P, Regu_N, T=T)

        ' filter output genes from cols (align with Y) '
        self.idx_list = output_idx_list if output_idx_list!=None\
                else list(range(self.Regu_P_0.shape[1]))

        ' initialize pos & neg KB '
        self.KB_P = self.Regu_P_0[:,self.idx_list]
        self.KB_N = self.Regu_N_0[:,self.idx_list]
        self.KB = torch.clamp(self.KB_P - self.KB_N, -1.,1.)


    def closure_(self, T=None, closure_type='naive'):
        '''
        Inplace & Nonstatic Ver of KB Closure

        Args
            T:
            closure_type:

        Return Values
            self.KB_P:
            self.KB_N:
            self.T:
        '''
        boolean = False if closure_type=='weighted' else True

        R_P, R_N, self.T = self.closure(self.Regu_P_0, self.Regu_N_0, T=T, boolean=boolean, device=self.device)
        R_P_2, R_N_2,_ = self.closure(self.Regu_P_0, self.Regu_N_0, T=2, boolean=boolean, device=self.device)
        self.KB_P, self.KB_N = R_P, R_N

        R_diff = R_P - R_N

        if closure_type == 'combined':
            self.KB = torch.where(R_P.bool() & R_N.bool(),
                         torch.where(R_P_2.bool()&R_N_2.bool(),
                                     torch.clamp(self.Regu_P_0-self.Regu_N_0,-1,1),
                                     torch.clamp(R_P_2-R_N_2,-1,1)), R_diff)

        elif closure_type == 'weighted':
            self.KB = torch.clamp(
                    torch.mul(torch.sign(R_diff),
                              torch.max(torch.abs(R_diff)-5,
                                        torch.zeros_like(R_diff))
                              ),-1,1)

        else:
            self.KB = torch.clamp(R_diff, -1,1)

        return self.KB_P, self.KB_N, self.T
    

    def violated(self,
                 Y: torch.Tensor,
                 X: torch.Tensor,
                 mask=None):
        '''
        violated count across all data points (matrix)
        Args:
            Y: 
            X:
            mask:
        '''

        deduction = torch.clamp((X @ self.KB)[:,self.idx_list], -1.,1.).int()
        if mask == None:
            vio_cnt = torch.count_nonzero(deduction != Y)
        else:
            vio_cnt = torch.count_nonzero((deduction != Y)[mask])
        return vio_cnt


    def deduce(self, X: torch.Tensor):
        ''' deduction result (multiplication) '''
        return torch.clamp((X @ self.KB)[:,self.idx_list], -1.,1.).int()
    

    def refine(self,
               X,
               Y,
               C=1,
               k=None,
               t=1,
               t0=100,
               epochs= 1000,
               lr= 1e-3,
               approx= 'exp',
               verbose=False):
        '''
        knowledge refinement via sparse learning

        Args
            self:
            X:
            Y:
            C:
            k:
            t0:
            t:
            epochs:
            lr:
            decay_rate:
            verbose:
        '''

        if k == None:
            k = self.T

        data = torch.clamp(X.T @ Y.float(), -1,1)
        Omega = torch.any((data!=0), axis=1)

        if approx == 'tanh':
            KB_opt, _ = self.sparse_opt_tanh(Y = data,
                                        X0 = self.Regu_0,
                                        Omega = Omega,
                                        label_set = self.idx_list,
                                        C = C,
                                        k = k,
                                        t = t,
                                        t0 = t0,
                                        lr = lr,
                                        epochs = epochs,
                                        verbose = verbose,
                                        device = self.device)

            KB_opt_k = torch.round(tanh_power(KB_opt, k, t))
            self.KB = torch.clamp(KB_opt_k, -1,1)

            self.KB_P = torch.clamp(self.KB, 0,1)
            self.KB_N = torch.clamp(-self.KB, 0,1)
            self.Regu_0 = torch.clamp(torch.round(KB_opt), -1,1)
            self.Regu_P_0 = torch.clamp(torch.round(KB_opt), 0,1)
            self.Regu_N_0 = torch.clamp(torch.round(-KB_opt), 0,1)

        else:
            KB_P_opt, KB_N_opt, _ = self.sparse_opt(Y = data,
                                        X0_P = self.Regu_P_0,
                                        X0_N = self.Regu_N_0,
                                        Omega = Omega,
                                        label_set = self.idx_list,
                                        C = C,
                                        k = k,
                                        t = t,
                                        t0 = t0,
                                        lr = lr,
                                        epochs = epochs,
                                        verbose = verbose,
                                        device = self.device)

            self.KB_P, self.KB_N = exp_closure(KB_P_opt, KB_N_opt, k, t)
            self.KB_P, self.KB_N = torch.round(self.KB_P), torch.round(self.KB_N)
            self.KB = torch.clamp(self.KB_P - self.KB_N, -1,1)

            self.Regu_P_0 = torch.clamp(torch.round(exp_soft(KB_P_opt, t0)), 0,1)
            self.Regu_N_0 = torch.clamp(torch.round(exp_soft(KB_N_opt, t0)), 0,1)
            self.Regu_0 = torch.clamp(torch.round(torch.where(KB_P_opt-KB_N_opt >= 0,
                exp_soft(KB_P_opt-KB_N_opt, t0), -exp_soft(KB_N_opt-KB_P_opt, t0))), -1,1)


    def save(self, path: str):
        coo_R0_P = coo_matrix(self.Regu_P_0.cpu().numpy())
        coo_R0_N = coo_matrix(self.Regu_N_0.cpu().numpy())
        coo_Rk_P = coo_matrix(self.KB_P.cpu().numpy())
        coo_Rk_N = coo_matrix(self.KB_N.cpu().numpy())
        coo_KB =   coo_matrix(self.KB.cpu().numpy())

        np.savez_compressed(path, 
                    data1=  coo_R0_P.data,
                    row1=   coo_R0_P.row,
                    col1=   coo_R0_P.col,
                    shape1= coo_R0_P.shape,
                    data2=  coo_R0_N.data,
                    row2=   coo_R0_N.row,
                    col2=   coo_R0_N.col,
                    shape2= coo_R0_N.shape,
                    data3=  coo_Rk_P.data,
                    row3=   coo_Rk_P.row,
                    col3=   coo_Rk_P.col,
                    shape3= coo_Rk_P.shape,
                    data4=  coo_Rk_N.data,
                    row4=   coo_Rk_N.row,
                    col4=   coo_Rk_N.col,
                    shape4= coo_Rk_N.shape,
                    data5=  coo_KB.data,
                    row5=   coo_KB.row,
                    col5=   coo_KB.col,
                    shape5= coo_KB.shape,)


    def load(self, path: str):
        loaded_data = np.load(path)

        coo_R0_P = coo_matrix((loaded_data['data1'],
                               (loaded_data['row1'], loaded_data['col1'])),
                               shape=tuple(loaded_data['shape1']))
        coo_R0_N = coo_matrix((loaded_data['data2'],
                               (loaded_data['row2'], loaded_data['col2'])),
                               shape=tuple(loaded_data['shape2']))
        coo_Rk_P = coo_matrix((loaded_data['data3'],
                               (loaded_data['row3'], loaded_data['col3'])),
                               shape=tuple(loaded_data['shape3']))
        coo_Rk_N = coo_matrix((loaded_data['data4'],
                               (loaded_data['row4'], loaded_data['col4'])),
                               shape=tuple(loaded_data['shape4']))
        coo_KB =   coo_matrix((loaded_data['data5'],
                               (loaded_data['row5'], loaded_data['col5'])),
                               shape=tuple(loaded_data['shape5']))

        self.Regu_P_0 = torch.tensor(coo_R0_P.toarray()).to(self.device)
        self.Regu_N_0 = torch.tensor(coo_R0_N.toarray()).to(self.device)
        self.KB_P = torch.tensor(coo_Rk_P.toarray()).to(self.device)
        self.KB_N = torch.tensor(coo_Rk_N.toarray()).to(self.device)
        self.KB = torch.tensor(coo_KB.toarray()).to(self.device)


    def eval(self):
        G = nx.from_numpy_array(\
                torch.clamp(torch.abs(self.Regu_P_0)+torch.abs(self.Regu_N_0), 0,1)\
                .cpu().numpy(), create_using=nx.DiGraph)
        #G = nx.from_numpy_array(self.Regu_0.cpu().numpy(), create_using=nx.DiGraph)

        # Basic stats
        #for k,G in {'R_0_P':G_p,'R_0_N':G_n}.items():#, 'R_k_P':G_k_p, 'R_k_N':G_k_n}.items():
        scores = {}
        scores['num_nodes'] = G.number_of_nodes()
        scores['num_edges'] = G.number_of_edges()
        scores['density'] = nx.density(G)
        
        # Degree-related
        scores['avg_degree'] = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
        #scores['avg_out_degree'] = sum(dict(G.out_degree()).values()) / G.number_of_nodes()
        
        # Clustering (need undirected projection)
        scores['avg_clustering'] = nx.average_clustering(G.to_undirected())
        
        # Path-based (if connected)
        #if nx.is_weakly_connected(G):
        #    UG = G.to_undirected()
        #    scores['avg_path_length'] = nx.average_shortest_path_length(UG)
        #    scores['diameter'] = nx.diameter(UG)
        #else:
        #    scores['avg_path_length'] = None
        #    scores['diameter'] = None
        
        # Assortativity
        scores['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
        
        # Modularity (via greedy community detection)
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G.to_undirected()))
        scores['modularity'] = nx.algorithms.community.quality.modularity(G.to_undirected(), communities)
        
        # Centrality
        bet = nx.betweenness_centrality(G)
        clo = nx.closeness_centrality(G)
        scores['avg_betweenness'] = np.mean(list(bet.values()))
        scores['avg_closeness'] = np.mean(list(clo.values()))
        
        ''' Print results '''
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")
        
        return scores



    @staticmethod
    def closure(R_P_0: torch.Tensor,
                R_N_0: torch.Tensor,
                T=None,
                boolean=True,
                device=torch.device('cpu')):
        '''
        Transitive Closure of Regulatory Matrix

        Args:
            R_P_0:
            R_N_0:
            T:
            boolean:
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
        
            if boolean:
                R_P, R_N = torch.clamp(R_P,0,1), torch.clamp(R_N,0,1)
            else:
                R_P /= torch.min(R_P[R_P!=0])
                R_N /= torch.min(R_N[R_N!=0])
        
            if torch.all(R_P == R_P_) and torch.all(R_N == R_N_):
                break
            cnt += 1
        
        R_N = torch.where(I.bool(), 0, R_N)
        return R_P, R_N, cnt


    @staticmethod
    def sparse_opt(Y,
                   X0_P,
                   X0_N,
                   Omega,
                   C,
                   k,
                   t,
                   t0,
                   label_set=None,
                   lr=1e-3,
                   epochs=1000,
                   tol=1e-3,
                   device=torch.device('cpu'),
                   verbose=False):
        """
        Optimization for Matrix Knowledge Refinement:
        min_X ||X^k - Y||_F^2 + C * ||X - X0||_F^2
    
        Args:
            Y: torch.Tensor, supervised data
            X0: torch.Tensor, original KB
            C: float, regularization strength
            k: int, power of X
            learning_rate: float, learning rate for gradient descent
            max_iter: int, maximum number of iterations
            tol: float, tolerance for convergence
            verbose: bool, whether to print progress
    
        Returns:
            X: torch.Tensor, the optimized matrix
            losses: list, loss values over iterations
        """

        if label_set == None:
            label_set = list(range(X0_P.shape[1]))
    
        X_P = X0_P.clone().detach().requires_grad_(True)
        X_P.to(device)
        X_N = X0_N.clone().detach().requires_grad_(True)
        X_N.to(device)
    
        # Use Adam optimizer for better convergence
        optimizer = torch.optim.Adam([X_P, X_N], lr=lr)
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
    
            Xk_P, Xk_N = exp_closure(X_P, X_N, k, t)
            Xk = Xk_P - Xk_N
            loss1 = torch.norm((Xk[:,label_set] - Y)[Omega], p='fro') ** 2

            loss2 = torch.norm(exp_soft(X_P, t0) - X0_P, p=1) + torch.norm(exp_soft(X_N, t0) - X0_N, p=1)
            #loss3 = torch.norm(exp_soft(X_P, t0) - X0_P, p='nuc') + torch.norm(exp_soft(X_N, t0) - X0_N, p='nuc')
            loss = loss1 + C * loss2 #+ C * loss3

            
            # Backpropagate
            loss.backward()
            optimizer.step()
    
            with torch.no_grad():
                X_P.data = X_P.data.clamp(min=0)
                X_N.data = X_N.data.clamp(min=0)
            
            losses.append(loss.item())
            
            #if epoch > 0 and abs(losses[-1] - losses[-2]) < tol:
            #    if verbose:
            #        print(f"Converged at iteration {epoch}")
            #    break
            
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                loss_round = torch.count_nonzero((torch.round(Xk[:,label_set])-Y)[Omega])

                f1 = f1_score(
                    torch.round(Xk[:,label_set][Omega]).flatten().detach().cpu().numpy(),
                    Y[Omega].flatten().detach().cpu().numpy(), average='macro')
    
                Xk_P_, Xk_N_, _ = RegulatoryKB.closure(torch.round(exp_soft(X_P,t0)), torch.round(exp_soft(X_N, t0)),T=k, device=device)
                Xk_ = torch.clamp(Xk_P_ - Xk_N_,-1,1)

                
                print(f"Iteration {epoch}: Loss = {loss.item():.6f}")
                print(f'|Xk-Y|_F: {loss1.item(): .6f}, |X-X0|: {loss2.item(): .6f}')
                #print(f'nuc: {loss3.item():.6f}')
                print(f'rounded |X_k-Y|_0 = {loss_round}, f1 = {f1: .6f}, approx slack: {torch.count_nonzero(Xk_ - Xk)}')

        return X_P.detach(), X_N.detach(), losses

    @staticmethod
    def sparse_opt_tanh(Y,
                   X0,
                   Omega,
                   C,
                   k,
                   t,
                   t0,
                   label_set=None,
                   lr=1e-3,
                   epochs=1000,
                   tol=1e-3,
                   device=torch.device('cpu'),
                   verbose=False):
        """
        Optimization for Matrix Knowledge Refinement:
        min_X ||X^k - Y||_F^2 + C * ||X - X0||_F^2
    
        Args:
            Y: torch.Tensor, supervised data
            X0: torch.Tensor, original KB
            C: float, regularization strength
            k: int, power of X
            learning_rate: float, learning rate for gradient descent
            max_iter: int, maximum number of iterations
            tol: float, tolerance for convergence
            verbose: bool, whether to print progress
    
        Returns:
            X: torch.Tensor, the optimized matrix
            losses: list, loss values over iterations
        """

        if label_set == None:
            label_set = list(range(X0.shape[1]))
    
        X = X0.clone().detach().requires_grad_(True)
        X.to(device)
    
        # Use Adam optimizer for better convergence
        optimizer = torch.optim.Adam([X], lr=lr)
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
    
            Xk = tanh_power(X, k, t)
            loss1 = torch.norm((Xk[:,label_set] - Y)[Omega], p='fro') ** 2

            loss2 = torch.norm(tanh_soft(X, t0) - X0, p=1)
            loss = loss1 + C * loss2

            
            # Backpropagate
            loss.backward()
            optimizer.step()
    
            with torch.no_grad():
                X.data = X.data.clamp(min=0)
            
            losses.append(loss.item())
            
            # Check for convergence
            if epoch > 0 and abs(losses[-1] - losses[-2]) < tol:
                if verbose:
                    print(f"Converged at iteration {epoch}")
                break
            
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                loss_round = torch.count_nonzero((torch.round(Xk[:,label_set])-Y)[Omega])

                f1 = f1_score(
                    torch.round(Xk[:,label_set][Omega]).flatten().detach().cpu().numpy(),
                    Y[Omega].flatten().detach().cpu().numpy(), average='macro')
    
                Xk_ = torch.clamp(torch.matrix_power(torch.round(tanh_soft(X,t0)),k),-1,1)
                loss_round_ = torch.count_nonzero((torch.round(Xk_)[:,label_set]-Y)[Omega])

                f1_ = f1_score(
                    torch.round(Xk_[:,label_set][Omega]).flatten().detach().cpu().numpy(),
                    Y[Omega].flatten().detach().cpu().numpy(), average='macro')
                
                print(f"Iteration {epoch}: Loss = {loss.item():.6f}")
                print(f'|Xk-Y|_F: {loss1.item(): .6f}, |X-X0|: {loss2.item(): .6f}')
                print(f'rounded |X_k-Y|_0 = {loss_round}, f1 = {f1: .6f}, approx slack: {torch.count_nonzero(Xk_ - torch.round(tanh_power(X,k,t)))}')
                print(f'rounded before pow |X_k-Y|_0 = {loss_round_}, f1 = {f1_: .6f}\n')

        return X.detach(), losses



if __name__ == '__main__':
    # NOTE tmp test

    
    regulatoryKB = RegulatoryKB(pos_trn_pth='rules/regu_pos.npz',
                                neg_trn_pth='rules/regu_neg.npz')

    print(regulatoryKB.KB_P, regulatoryKB.KB_P.shape)
    print('closure times:', regulatoryKB.T)

    print(np.count_nonzero(np.sum(regulatoryKB.KB_P & regulatoryKB.KB_N, axis=0)))
    print(np.count_nonzero(np.sum(regulatoryKB.KB_P | regulatoryKB.KB_N, axis=0)))

    #metabolicKB = MetabolicKB(pos_gem_pth='rules/gem_pos.npz', neg_gem_pth='rules/gem_neg.npz', annotation_pth='rules/gem_annot.npz')
    #print(metabolicKB.KB_P, metabolicKB.KB_P.shape)
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
