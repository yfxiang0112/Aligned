import numpy as np
import pygraphblas as pgb
from zoopt import Dimension, Objective, Parameter, Opt, Solution

n_rows, n_cols = 5,5
max_modify = 1

def xor(M1: pgb.Matrix, M2: pgb.Matrix) -> pgb.Matrix:
    ''' M2 as the smaller matrix '''
    assert M1.shape == M2.shape
    idx_1 = M1.to_lists()
    idx_1= set(zip(idx_1[0], idx_1[1]))
    idx_2 = M2.to_lists()

    for i,j in zip(idx_2[0], idx_2[1]):
        if (i,j) in idx_1:
            idx_1.remove((i,j))
        else:
            idx_1.add((i,j))
    row, col = zip(*list(idx_1))
    return pgb.Matrix.from_lists(row,col, nrows=M1.shape[0], ncols=M1.shape[1], typ=pgb.BOOL)

def optvec2pgb(x, n_rows, n_cols, max_modify):
    ''' optimization vector to I_pos and I_neg '''
    x_values = x.get_x()
    x_pos = x_values[:n_rows*max_modify]
    x_neg = x_values[n_rows*max_modify:]

    def sol2pgb(x):
        #print(x)
        return pgb.Matrix.from_lists(
                    [ i // max_modify   for i in range(n_rows * max_modify) if x[i]!=-1],  # Row indices
                    [ x[i]              for i in range(n_rows * max_modify) if x[i]!=-1],   # Column indices
                    #V=x,                                      # Values (0 or 1)
                    nrows=n_rows, ncols=n_cols,
                    typ=pgb.BOOL )
    return sol2pgb(x_pos), sol2pgb(x_neg)

####################


def constraint(X1, X2, constant):
    """
    Check if the constraints ||X1||_inf <= constant and ||X2||_inf <= constant are satisfied.
    X1: A pygraphblas.Matrix with elements 0 or 1.
    X2: A pygraphblas.Matrix with elements 0 or 1.
    constant: The constraint constant.
    """
    # Compute the infinity norm of each row for X1 and X2
    row_norms_X1 = X1.reduce_vector(cast=pgb.INT32)  # Reduce each row of X1 using the max monoid
    row_norms_X2 = X2.reduce_vector(cast=pgb.INT32)  # Reduce each row of X2 using the max monoid
    #print(X1)
    #print(row_norms_X1)
    #exit()

    # Check if all row norms satisfy the constraint for both matrices
    return all(val <= constant for val in row_norms_X1.vals) and all(val <= constant for val in row_norms_X2.vals)

def violation(X1,X2):
    """
    Compute the value of the objective function.
    X: A matrix with elements 0 or 1.
    """
    # This is a placeholder function. Replace it with your actual objective function.
    ref1 = pgb.Matrix.from_lists([0,3,4,4,4,4,4], [1,2,0,1,2,3,4], nrows=X1.shape[0], ncols=X1.shape[1], typ=pgb.BOOL)
    ref2 = pgb.Matrix.from_lists([0,1,2,2,4], [1,2,4,1,3], nrows=X1.shape[0], ncols=X1.shape[1], typ=pgb.BOOL)
    #print(ref1)
    #print(X1)
    #print(xor(X1,ref1))
    return xor(X1, ref1).reduce_int() + xor(X2, ref2).reduce_int()  # Example: Return the sum of the matrix elements

def objective(x):
    X1, X2 = optvec2pgb(x, n_rows, n_cols, max_modify)

    # Check if the constraints are satisfied
    if not constraint(X1, X2, max_modify):
        return float('inf')  # Return infinity if constraints are violated
    return violation(X1, X2)  # Otherwise, return the objective function value


from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt
#opt_dim = Dimension(size= n_rows * n_cols * 2,
#                    regs= [[0, 1]] * (n_rows * n_cols * 2),
#                    tys= [False] * (n_rows * n_cols) * 2)
opt_dim = Dimension(size= n_rows * max_modify * 2,
                    regs= [[-1, n_cols-1]] * (n_rows * max_modify * 2),
                    tys= [False] * (n_rows * max_modify * 2))
opt_obj = Objective(objective, opt_dim)
solution = Opt.min(opt_obj, Parameter(budget=1000))
X1, X2 = optvec2pgb(solution, n_rows, n_cols, max_modify)
print(X1)
print(X2)
