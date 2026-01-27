import torch
import torch.nn as nn
import torch.optim as optim
from cobra.io import load_model
import pandas as pd
import numpy as np
import time

def solve_lp(c, A, lb, ub, max_iter=100, lr=0.01, tol=1e-6, device='cuda'):
    """
    Solve linear programming problem:
        minimize c^T x
        subject to Ax = 0
                   lb <= x <= ub
    
    Args:
        c: cost vector (n,)
        A: equality constraint matrix (m, n)
        lb: lower bounds (n,)
        ub: upper bounds (n,)
        max_iter: maximum iterations
        lr: learning rate
        tol: convergence tolerance
        device: 'cuda' or 'cpu'
    """
    # Move everything to GPU
    c = c.to(device)
    A = A.to(device)
    lb = lb.to(device)
    ub = ub.to(device)
    
    # Initialize variables
    n = c.shape[0]
    x = nn.Parameter((lb + ub) / 2).to(device)
    
    # Optimizer
    optimizer = optim.Adam([x], lr=lr)
    
    # Projection function for bounds
    def project_bounds(x):
        return torch.clamp(x, lb, ub)
    
    # Projection function for equality constraints (Ax=0)
    def project_equality(x):
        # Using pseudo-inverse for projection: x = (I - A^T (AA^T)^-1 A) x
        if A.shape[0] > 0:  # if there are equality constraints
            A_pinv = torch.pinverse(A)
            return x - A_pinv @ (A @ x)
        return x
    
    # Optimization loop
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Objective
        loss = c @ x
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Projection steps
        with torch.no_grad():
            # Project onto bounds
            x.data = project_bounds(x.data)
            
            # Project onto equality constraints
            x.data = project_equality(x.data)
        
        # Check convergence
        if i > 0 and i % 100 == 0:
            constraint_violation = torch.norm(A @ x, p=2) if A.shape[0] > 0 else 0
            if constraint_violation < tol and torch.all(x >= lb - tol) and torch.all(x <= ub + tol):
                break
    
    return x.detach()







if __name__ == "__main__":



    # Load the iML1515 model
    # First try to load from local models, if not found, load from BiGG database
    try:
        model = load_model("iML1515")
    except:
        print("Local model not found, downloading from BiGG database...")
        model = cobra.io.load_json_model("http://bigg.ucsd.edu/static/models/iML1515.json")
    
    # Get the stoichiometric matrix (S matrix)
    # Create a pandas DataFrame for better visualization
    reaction_ids = [r.id for r in model.reactions]
    metabolite_ids = [m.id for m in model.metabolites]
    
    S_matrix = pd.DataFrame(
        data=0,
        index=metabolite_ids,
        columns=reaction_ids,
        dtype=float
    )
    
    # Fill in the stoichiometric coefficients
    for reaction in model.reactions:
        for metabolite, coeff in reaction.metabolites.items():
            S_matrix.at[metabolite.id, reaction.id] = coeff
    
    # Get the objective function
    objective_reactions = [r.id for r in model.reactions if r.objective_coefficient != 0]
    objective_coefficients = {r.id: r.objective_coefficient for r in model.reactions if r.objective_coefficient != 0}
    
    # Print some information
    print("Model loaded:", model.id)
    print("Number of reactions:", len(model.reactions))
    print("Number of metabolites:", len(model.metabolites))
    print("\nObjective function:")
    print(" + ".join([f"{coeff}*{rxn}" for rxn, coeff in objective_coefficients.items()]))
    
    # Optionally save the S matrix to a CSV file
    #S_matrix.to_csv("iML1515_stoichiometric_matrix.csv")
    
    # Show the first few rows and columns of the S matrix
    print("\nPreview of the stoichiometric matrix:")
    print(S_matrix)

    objective = S_matrix.columns.isin(objective_coefficients).astype(np.float32)
    S_matrix = S_matrix.to_numpy().astype(np.float32)

    print(np.nonzero(objective))


    # Problem dimensions
    n = S_matrix.shape[1]  # variables
    m = S_matrix.shape[0]   # equality constraints
    
    # Generate random problem
    c = torch.tensor(objective)
    A = torch.tensor(S_matrix)
    lb = torch.tensor([-1000]*n)
    ub = torch.tensor([1000]*n)
    
    # Solve on GPU
    t_0 = time.time()
    solution = solve_lp(-c, A, lb, ub)
    t = time.time() - t_0

    print(f'GPU solution: {solution}\nObjective: {c.to("cpu") @ solution.to("cpu")}\nTime: {t}\n')
    
    #print("Solution:", solution)
    #print("Objective value:", c.to('cpu') @ solution.to('cpu'))
    #print("Equality constraint violation:", torch.norm(A @ solution, p=2))
    #print(f'Solution time: {t}')
    #print("Bounds satisfaction:", torch.all(solution >= lb - 1e-5), torch.all(solution <= ub + 1e-5))

    t_0 = time.time()
    solution = model.optimize()
    t = time.time() - t_0
    print(f'cobra solution: {solution.fluxes}\nObjective: {solution.objective_value}\ntime: {t}')
