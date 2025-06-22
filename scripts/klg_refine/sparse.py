import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def print_log(s: str):
    with open('out.txt', 'a') as f:
        f.write(s+'\n')

def exp_soft(X,t):
    return 1 - torch.exp(-t * X)

def exp_power(X, k, t):
    #return torch.matrix_power(1 - torch.exp(-t * X), k)
    return 1 - torch.exp(-t * torch.matrix_power(X, k))

def tanh_power(X, k):
    #return torch.matrix_power(1 - torch.exp(-t * X), k)
    return 1 - torch.tanh(torch.matrix_power(X, k))


def optimize_X(Y, X0, Omega, C, k, t, init_lr=1e-3, max_iter=1000, tol=1e-4, decay_rate=0.995, device=torch.device('cpu'), verbose=False):
    """
    Solve the optimization problem:
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
    # Initialize X with X0 (requires gradient)
    X = X0.clone().detach().requires_grad_(True)
    #X = torch.zeros_like(X0).requires_grad_(True)
    X.to(device)

    # Use Adam optimizer for better convergence
    optimizer = torch.optim.Adam([X], lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    
    losses = []
    for epoch in range(max_iter):
        optimizer.zero_grad()

        #rounded = X.round()
        #X = X + (rounded - X).detach()
        
        # Compute X^k
        #Xk = torch.matrix_power(X, k)
        
        # Compute the two terms of the loss

        #C = 1e-7 if epoch<50 else .1
        #Xk_tmp = torch.matrix_power(X,k)
        #print(Xk_tmp)
        #print(torch.max(Xk_tmp), torch.min(Xk_tmp[Xk_tmp!=0]))
        #print()

        Xk = exp_power(X, k, t)
        loss1 = torch.norm((Xk - Y)[Omega], p='fro') ** 2
        #loss1 = torch.norm((tanh_power(X, k) - Y)[Omega], p='fro') ** 2
        loss2 = torch.norm(exp_soft(X, t) - X0, p=1)# ** 2
        loss = loss1 + C * loss2
        loss_round = torch.count_nonzero((torch.round(Xk)-Y)[Omega])
        f1 = f1_score(
                torch.round(Xk[Omega]).flatten().detach().cpu().numpy(),
                Y[Omega].flatten().detach().cpu().numpy())
        #loss_round = torch.count_nonzero((torch.round(tanh_power(X,k))-Y)[Omega])
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            X.data = X.data.clamp(min=0)
        
        losses.append(loss.item())
        
        # Check for convergence
        if epoch > 0 and abs(losses[-1] - losses[-2]) < tol:
            if verbose:
                print(f"Converged at iteration {epoch}")
            break
        
        #if verbose and (epoch % 20 == 0 or epoch == max_iter - 1):
        print(f"Iteration {epoch}: Loss = {loss.item():.6f}")
        print(f'|Xk-Y|_F: {loss1.item(): .6f}, |X-X0|: {loss2.item(): .6f}')
        print(f'rounded |X_k-Y|_0 = {loss_round}, f1 = {f1: .6f}\n')
    
    return X.detach(), losses

# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)
    
    # Define problem dimensions
    n = 100  # Matrix size (n x n)
    k = 5  # Power of X
    t = 1 # Coeff in exp surrogate func
    C = .1  # Regularization strength
    
    ## Generate random Y and X0
    #X0 = torch.tensor(np.random.choice([0., 1.], size=(n,n), p=[.9, .1]).astype(np.float32))
    #Y = torch.tensor(np.random.choice([0., 1.], size=(n,n), p=[.6, .4]).astype(np.float32))
    #print(f'init loss = {torch.norm(torch.clip(torch.matrix_power(X0, k),0,1)-Y,p="fro")}')
    #
    #print("Y =", Y)
    #print("X0 =", X0)

    #######################################

    from scipy.sparse import load_npz
    import pandas as pd

    pos_regu = load_npz('rules/regu_pos.npz').toarray()
    neg_regu = load_npz('rules/regu_neg.npz').toarray()
    connectv_kb = np.clip(np.abs(pos_regu) + np.abs(neg_regu), 0,1)
    
    ''' mat mul X (pert) -> Y & abs: get connectivity matrix
        (as supervision for mat compl) '''
    X_l = np.load('dataset/precise1k/X_label.npy')
    Y_l = np.load('dataset/precise1k/Y_label.npy')
    connectv_sup = np.clip(np.abs(X_l).T @ np.abs(Y_l), 0,1)
    
    ''' align with precise1k genome '''
    gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
    idx_map = [idx for idx,v in enumerate(gene_idx['precise1k_idx']) if v != -1]
    connectv_kb = connectv_kb[idx_map][:,idx_map]
    connectv_sup = connectv_sup[idx_map][:,[i for i in gene_idx['precise1k_idx'] if i != -1]]

    #Omega = np.any((connectv_kb!=0) & (connectv_sup!=0), axis=1)
    Omega = np.any((connectv_sup!=0), axis=1)

    Y = torch.tensor(connectv_sup)
    X0 = torch.tensor(connectv_kb)
    Omega = torch.tensor(Omega)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    Y.to(device)
    X0.to(device)
    Omega.to(device)

    print(f'check X0^k == 1-exp(X0^k): {torch.count_nonzero(torch.clamp(torch.matrix_power(X0,k),0,1) - torch.round(exp_power(X0,k,t))) == 0}')
    print(f'check X0^k == tanh(X0^k): {torch.count_nonzero(torch.clamp(torch.matrix_power(X0,k),0,1) - torch.round(tanh_power(X0,k))) == 0}')
    
    #print(f'init loss = {torch.norm((torch.clip(torch.matrix_power(X0, k),0,1)-Y)[Omega],p="fro") ** 2}')
    print(f'init Xk-Y diff = {torch.count_nonzero((torch.clip(torch.matrix_power(X0, k),0,1)-Y)[Omega])}')
    print(f'init f1 = {f1_score(torch.clip(torch.matrix_power(X0, k),0,1)[Omega].flatten().detach().cpu().numpy(), Y[Omega].flatten().detach().cpu().numpy())}')

    #######################################
    
    # Solve the optimization problem
    X_opt, losses = optimize_X(Y, X0, Omega, C, k, t, init_lr=1e-3, decay_rate=.995, max_iter=1000, device=device, verbose=True)
    
    #X_opt = torch.round(X_opt.clip(0,1))
    X_opt_k = exp_power(X_opt, k, t)
    print(f"\nOptimized X = {X_opt}")
    print(f"X^k = {X_opt_k}")
    print(f"Y = {Y}")
    print(f"Frobenius norm of X^k - Y: {torch.norm((X_opt_k - Y)[Omega], p='fro').item()}")
    print(f"L1 norm of X - X0:, {torch.norm(X_opt - X0, p=1).item()}")

    print(f'X_k rounded = \n{torch.round(X_opt_k)}')
    print(f'rounded X_k-Y diff = {torch.count_nonzero((torch.round(X_opt_k)-Y)[Omega])}')

    X_opt_round = torch.round(exp_soft(X_opt, t))
    print(f'X rounded = \n{X_opt_round}')
    print(f'|X-X0|_1 = {torch.norm(X_opt_round-X0, p=1)}')
