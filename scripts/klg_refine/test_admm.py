import numpy as np
from scipy.linalg import svd
from typing import Tuple

def mat_power(X: np.ndarray, k: int) -> np.ndarray:
    """计算矩阵的k次幂（通过连乘）"""
    result = np.eye(X.shape[0])
    for _ in range(k):
        result = np.clip(result @ X, 0.,1.)
    return result

def mat_mul(X_1:np.ndarray, X_2:np.ndarray) -> np.ndarray:
    return np.clip(X_1 @ X_2, 0.,1.)

def mat_plus(X_1: np.ndarray, X_2: np.ndarray) -> np.ndarray:
    return np.clip(X_1+X_2, 0.,1.)



def mat_mul_direc(X_1:np.ndarray, X_2:np.ndarray) -> np.ndarray:
    return np.clip(X_1 @ X_2, -1.,1.)

def mat_minus(X_1: np.ndarray, X_2: np.ndarray) -> np.ndarray:
    return np.clip(X_1-X_2, -1.,1.)


##===============================================##

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """L1范数的软阈值算子"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def nuclear_prox(X: np.ndarray, tau: float) -> np.ndarray:
    """核范数的近端算子（奇异值阈值）"""
    U, s, Vh = svd(X, full_matrices=False)
    s_thresh = soft_threshold(s, tau)
    #s_thresh = np.maximum(s - tau, 0)
    return U @ np.diag(s_thresh) @ Vh

def solve_X_subproblem(Y: np.ndarray, 
                      Z: np.ndarray, 
                      U: np.ndarray, 
                      X_init: np.ndarray,
                      rho: float, 
                      k: int,
                      max_iter: int = 1,
                      lr: float = 0.01) -> np.ndarray:
    """
    近似求解X子问题：
    min_X ||X-Y||_* + (rho/2)||Z - X^k + U||_F^2
    使用梯度下降+近端梯度法
    """
    X = X_init.copy()
    for _ in range(max_iter):
        # 计算当前X^k和残差
        Xk = mat_power(X, k)
        residual = Xk - Z - U
        print(f'U: {np.count_nonzero(U)}\n{U}\n\nZ: {np.count_nonzero(Z)}\n{Z}\n\nresidual: {np.count_nonzero(residual)}\n{residual}\n')
        print(f'Xk: {np.count_nonzero(Xk)}\n{Xk}\n')
        
        # 计算梯度 (需推导线性化梯度)
        grad = np.zeros_like(X)
        for i in range(1, k+1):
            grad += mat_mul_direc(mat_mul_direc(mat_power(X, i-1).T , residual) , mat_power(X, k-i).T)
        grad /= (k)
        #print(f'grad: {np.count_nonzero(grad)}\n{grad}\n')
        
        # 梯度下降步（注意：这里简化处理，实际需要更精确的线性化）
        X_temp = X - lr * rho * grad
        
        # 近端算子处理核范数项
        X = nuclear_prox(X_temp - Y, lr) + Y
        
    return X

def robust_matrix_completion(Y: np.ndarray,
                            A: np.ndarray,
                            Omega: np.ndarray,
                            k: int,
                            C: float,
                            rho: float = 1.,
                            max_iter: int = 50,
                            tol: float = 1e-5) -> Tuple[np.ndarray, dict]:
    """
    主ADMM算法求解：
    min_X ||X-Y||_* + C||P_Omega(X^k) - P_Omega(A)||_1
    
    参数:
        Y: 目标矩阵 (n x n)
        A: 观测矩阵 (n x n)
        Omega: 观测位置掩码 (n x n bool数组)
        k: 矩阵幂次数
        C: L1惩罚系数
        rho: ADMM惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛容忍度
    
    返回:
        X: 优化后的矩阵
        info: 包含收敛信息的字典
    """
    n = Y.shape[0]
    X = Y.copy()  # 初始化
    Z = np.zeros_like(Y)
    U = np.zeros_like(Y)  # 对偶变量
    
    # 记录收敛信息
    primal_residuals = []
    dual_residuals = []
    obj_values = []
    
    iters = 0
    for iters in range(max_iter):
        X_prev = X.copy()
        
        
        # --- X更新 ---
        X = solve_X_subproblem(Y, Z, U, X, rho, k)
        
        # --- Z更新 ---
        Xk = mat_power(X, k)
        temp = Xk - U
        
        # 对观测位置应用软阈值，非观测位置直接赋值
        Z = np.where(Omega, 
                     soft_threshold(temp - A, C/rho) + A,
                     temp)
        print(f'temp:\n{temp}\n\nZ:\n{Z}\nA-Z: {np.count_nonzero(A-Z)}')
        
        # --- 对偶变量更新 ---
        U = U + (Z - Xk)
        #U = U + Xk - Z
        
        # --- 计算收敛性指标 ---
        primal_res = np.linalg.norm(Z - Xk, 'fro')
        dual_res = rho * np.linalg.norm(Xk - mat_power(X_prev, k), 'fro')
        
        # 计算目标函数值
        nuclear_norm = np.sum(np.abs(svd(X - Y, compute_uv=False)))
        l1_penalty = np.sum(np.abs((Xk - A)[Omega]))
        obj_value = nuclear_norm + C * l1_penalty
        
        # 记录信息
        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)
        obj_values.append(obj_value)

        print(f'{iters}th iter, |X-Y|_* = {nuclear_norm:.5f}, |Xk-A|_1 = {l1_penalty:.5f}, obj = {obj_value:.5f}')
        #print(f'result:\n{X}')

        if iters > 1:
            exit()
        
        # 检查收敛
        if primal_res < tol and dual_res < tol:
            print(f"在迭代 {iters} 次后收敛")
            break
            
    info = {
        'primal_residuals': primal_residuals,
        'dual_residuals': dual_residuals,
        'obj_values': obj_values,
        'iterations': iters + 1
    }
    
    return X, info

# 测试用例
if __name__ == "__main__":
    np.random.seed(42)
    n = 50  # 矩阵维度
    #rank = 5  # 真实矩阵的秩
    k = 3  # 矩阵幂次
    
    # 生成低秩矩阵Y
    #U = abs(np.random.randn(n, rank))
    #V = abs(np.random.randn(n, rank))
    #Y = U @ V.T
    Y = np.random.choice([0., 1.], size=(n,n), p=[.9, .1]).astype(np.float32)

    # 生成观测矩阵A (Y^k加噪声)
    A_clean = mat_power(Y, k)
    noise = np.random.choice([0., 1., -1.], size=(n,n), p=[.4, .3, .3]).astype(np.float32)
    #noise = n(0.1 * np.random.randn(n, n))
    A = mat_plus(A_clean , noise)
    
    # 随机采样观测位置 (50% 观测)
    #Omega = np.random.rand(n, n) < 0.5
    Omega = np.full(shape= (n,n), fill_value= True)

    print(f'init l1: {np.sum(np.abs((mat_power(Y,k) - A)[Omega]))}')
    
    # 运行算法
    X_est, info = robust_matrix_completion(Y, A, Omega, k=k, C=10)
    
    # 评估结果
    print(f"核范数项: {info['obj_values'][-1]}")
    print(f"最终原始残差: {info['primal_residuals'][-1]:.4f}")
    print(f'\nresult:\n{X_est}\n{np.round(X_est)}\n{mat_power(np.round(X_est),k)}\n\ninit:\n{Y}\n\ndata:\n{A}')
