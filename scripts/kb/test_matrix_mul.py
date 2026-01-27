import cupy as cp
import cupyx
from cupyx.scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.sparse import save_npz, load_npz
import numpy as np
import scipy.sparse as sp
import time
import torch

'''
Compare sparse float32 matrix-vector multiplication performance:
- CuPy sparse CSR (GPU)
- CuPy sparse COO (GPU)
- NumPy dense (CPU)
- SciPy sparse CSR (CPU)
'''

np.random.seed(42)

cp.cuda.Device(0).use()

pos = load_npz('rules/gem_pos.npz')
annot = load_npz('rules/gem_annot.npz')
A = (annot @ pos).T
A.data = np.clip(A.data, 0, 1)
A = A.astype(np.float32)

# Matrix and vector dimensions
m, n = A.shape[0], A.shape[1]
batch = 623 * 128

# SciPy COO and CSR matrices
A_cpu_coo = A.astype(np.float32)
A_cpu_csr = A_cpu_coo.tocsr()
A_cpu_csc = A_cpu_coo.tocsc()

# NumPy dense matrix
A_cpu_dense = A_cpu_csr.toarray()

# Float16 vector
B_cpu = np.random.choice([0., 1.], size=(n,batch), p=[.8, .2]).astype(np.float32)

# Convert to CuPy
A_gpu_csr = csr_matrix(A_cpu_csr)
A_gpu_csc = csc_matrix(A_cpu_csc)
A_gpu_coo = coo_matrix(A_cpu_coo)
A_gpu_dense = A_gpu_coo.toarray()
B_gpu = cp.array(B_cpu, dtype=cp.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_torch = torch.tensor(A_cpu_dense).to(device)
B_torch = torch.tensor(B_cpu).to(device)


# Warm-up GPU
cp.empty((1,)).sum()

# --- NumPy Dense (CPU) ---
start = time.perf_counter()
result_dense = A_cpu_dense @ B_cpu
print("NumPy Dense time:", time.perf_counter() - start)

# --- SciPy Sparse CSR (CPU) ---
start = time.perf_counter()
result_sparse_cpu = A_cpu_csr @ B_cpu
print("SciPy CSR time:", time.perf_counter() - start)

# --- SciPy Sparse COO (CPU) ---
start = time.perf_counter()
result_sparse_cpu = A_cpu_coo @ B_cpu
print("SciPy COO time:", time.perf_counter() - start)

# --- CuPy Sparse CSR ---
start = time.perf_counter()
result_gpu_csr = A_gpu_csr @ B_gpu
#cp.cuda.Device(0).synchronize()
print("CuPy CSR time:", time.perf_counter() - start)

# --- CuPy Sparse CSC ---
start = time.perf_counter()
result_gpu_csr = A_gpu_csc @ B_gpu
#cp.cuda.Device(0).synchronize()
print("CuPy CSC time:", time.perf_counter() - start)

# --- CuPy Sparse COO ---
start = time.perf_counter()
result_gpu_coo = A_gpu_coo @ B_gpu
#cp.cuda.Device(0).synchronize()
print("CuPy COO time:", time.perf_counter() - start)

# --- CuPy dense ---
start = time.perf_counter()
result_gpu_dense = A_gpu_dense @ B_gpu
#cp.cuda.Device(0).synchronize()
print("CuPy dense time:", time.perf_counter() - start)

# --- torch dense ---
start = time.perf_counter()
result_torch = A_torch @ B_torch
print("torch dense time:", time.perf_counter() - start)

# --- NumPy clip time ---
start = time.perf_counter()
result_clip_cpu = np.clip(result_dense, 0.,1.)
print("\nNumPy clip time:", time.perf_counter() - start)

# --- cupy clip time ---
start = time.perf_counter()
result_clip_gpu = cp.clip(result_gpu_dense, 0.,1.)
#cp.cuda.Device(0).synchronize()
print("CuPy clip time:", time.perf_counter() - start)

# --- torch clip time ---
start = time.perf_counter()
result_clip_gpu = torch.clamp(result_torch, 0.,1.)
#result_clip_gpu = result_torch.type(torch.BoolTensor)
print("torch clip time:", time.perf_counter() - start)

# --- Check correctness ---
assert np.allclose(result_dense, result_sparse_cpu, atol=1e-2), "Dense vs SciPy CSR mismatch"
assert np.allclose(result_dense, cp.asnumpy(result_gpu_csr), atol=1e-2), "Dense vs CuPy CSR mismatch"
assert np.allclose(result_dense, cp.asnumpy(result_gpu_coo), atol=1e-2), "Dense vs CuPy COO mismatch"
assert np.allclose(result_dense, cp.asnumpy(result_gpu_dense), atol=1e-2), "Dense vs CuPy dense mismatch"
assert np.allclose(result_dense, result_torch.to('cpu').numpy(), atol=1e-2), "Dense vs torch dense mismatch"
print("All methods produce approximately the same result.")
