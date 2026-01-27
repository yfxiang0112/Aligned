import time
import numpy as np
import cupy as cp
import torch

# Matrix dimensions
n = 2048  # Large enough to see GPU benefits

def benchmark_operations(name, op, *args):
    start = time.time()
    result = op(*args)
    if type(result)==torch.Tensor:  # For GPU tensors/arrays
        result = result.cpu()
    elif type(result)==np.ndarray:
        pass
    else:
        result.device.synchronize()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.6f} seconds")
    return elapsed

# Generate random boolean matrices
np_A = np.random.rand(n, n) > 0.5
np_B = np.random.rand(n, n) > 0.5

# CuPy arrays
cp_A = cp.array(np_A)
cp_B = cp.array(np_B)

# PyTorch tensors (on GPU)
device = 'cuda'
torch_A = torch.tensor(np_A, dtype=torch.float32, device=device)
torch_B = torch.tensor(np_B, dtype=torch.float32, device=device)

# ===== Boolean Operations =====
print(f"\nBenchmarking {n}x{n} matrices:")

# NumPy CPU
n_and = benchmark_operations("NumPy AND", np.logical_and, np_A, np_B)
n_or = benchmark_operations("NumPy OR", np.logical_or, np_A, np_B)
n_not = benchmark_operations("NumPy NOT", np.logical_not, np_A)

# CuPy GPU
c_and = benchmark_operations("CuPy AND", cp.logical_and, cp_A, cp_B)
c_or = benchmark_operations("CuPy OR", cp.logical_or, cp_A, cp_B)
c_not = benchmark_operations("CuPy NOT", cp.logical_not, cp_A)

# PyTorch GPU (using float tensors with clamping)
def torch_bool_op(a, b, op):
    # Convert to float, clamp to [0,1], then round to get boolean behavior
    a_float = a.float()
    b_float = b.float()
    res = op(a_float, b_float).clamp(0, 1).round().bool()
    return res

t_and = benchmark_operations("Torch AND", torch_bool_op, torch_A, torch_B, torch.mul)
t_or = benchmark_operations("Torch OR", torch_bool_op, torch_A, torch_B, lambda x,y: 1-(1-x)*(1-y))
t_not = benchmark_operations("Torch NOT", lambda x: (1 - x.float()).clamp(0,1).round().bool(), torch_A)

# ===== Matrix Multiplication =====
# NumPy (boolean to int)
n_matmul = benchmark_operations("NumPy MatMul", np.dot, np_A, np_B)

# CuPy (boolean to int)
c_matmul = benchmark_operations("CuPy MatMul", cp.dot, cp_A, cp_B)

# PyTorch (using float matmul with clamping)
def torch_bool_matmul(a, b):
    return (a.float() @ b.float()).clamp(0, 1).round().bool()

t_matmul = benchmark_operations("Torch MatMul", torch_bool_matmul, torch_A, torch_B)

# ===== Results Summary =====
print("\nSpeedup (GPU vs CPU):")
print(f"AND:    {n_and/c_and:.1f}x (CuPy), {n_and/t_and:.1f}x (Torch)")
print(f"OR:     {n_or/c_or:.1f}x (CuPy), {n_or/t_or:.1f}x (Torch)")
print(f"NOT:    {n_not/c_not:.1f}x (CuPy), {n_not/t_not:.1f}x (Torch)")
print(f"MatMul: {n_matmul/c_matmul:.1f}x (CuPy), {n_matmul/t_matmul:.1f}x (Torch)")
