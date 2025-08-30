import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from scipy.sparse import load_npz
import torch

from egoal.reasoner import RegualtoryKB


def create_consistency_matrix(Y1, Y2):
    """
    Create consistency matrix Yc where:
    Yc_{ij} = 0 if Y1_{ij} == Y2_{ij}
    Yc_{ij} = -1 if Y1_{ij} == 0 and Y2_{ij} != 0
    Yc_{ij} = 1 otherwise
    """
    Yc = np.zeros_like(Y1, dtype=int)
    
    # Case 1: Y1_{ij} == Y2_{ij}
    equal_mask = (Y1 == Y2)
    Yc[equal_mask] = 0
    
    # Case 2: Y1_{ij} == 0 and Y2_{ij} != 0
    zero_mask = (Y1 == 0) & (Y2 != 0)
    Yc[zero_mask] = -1
    
    # Case 3: All other cases (Y1_{ij} != Y2_{ij} and not both zero)
    other_mask = (~equal_mask) & (~zero_mask)
    Yc[other_mask] = 1
    
    return Yc

def create_adjacency_matrix(X, Yc, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create adjacency matrix A using PyTorch for GPU acceleration:
    A_{ij} = sum_{k: X_{ki} != 0} Yc_{kj}
    
    Optimized using matrix operations instead of loops.
    """
    # Convert to PyTorch tensors and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Yc_tensor = torch.tensor(Yc, dtype=torch.float32, device=device)
    
    # Create binary mask for non-zero elements in X
    # Shape: (m, n)
    X_mask = (X_tensor != 0).float()
    
    # Matrix multiplication: A = X_mask.T @ Yc
    # This computes: A[i,j] = sum_k (X_mask[k,i] * Yc[k,j])
    # Which is exactly: sum_{k: X[k,i] != 0} Yc[k,j]
    A = torch.mm(X_mask.T, Yc_tensor)
    
    # Convert back to numpy if needed
    return A.cpu().numpy()

def plot_network(A, title="Network Visualization"):
    """
    Plot network with A as adjacency matrix, colored by values in A
    """
    # Create graph and remove zero-weight edges efficiently
    G = nx.Graph()
    
    # Only add edges with non-zero weights
    rows, cols = np.where(A != 0)
    for i, j in zip(rows, cols):
        if i < j:  # Avoid duplicate edges for undirected graph
            weight = A[i, j]
            G.add_edge(i, j, weight=weight)
    
    
    # Create node colors based on A values
    node_colors = []
    for i in range(A.shape[0]):
        # Get the sum of weights for each node (diagonal represents self-connections)
        node_value = A[i, i]  # Using diagonal value for node coloring
        
        if node_value == 0:
            node_colors.append('green')
        elif node_value > 0:
            node_colors.append('red')
        else:
            node_colors.append('orange')
    
    # Create edge colors based on A values
    edge_colors = []
    edge_weights = []
    for u, v, data in G.edges(data=True):
        weight = A[u, v]
        edge_weights.append(abs(weight) / max(1, np.max(np.abs(A))))  # Normalize for visibility
        
        if weight == 0:
            edge_colors.append('green')
        elif weight > 0:
            edge_colors.append('red')
        else:
            edge_colors.append('orange')
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.8)
    
    # Draw edges with varying widths based on weight
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                          width=[w * 5 + 0.5 for w in edge_weights],  # Scale for visibility
                          alpha=0.6)
    
    # Optional: draw labels (can be commented out for large networks)
    # nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='green', label='A_{ij} = 0'),
        mpatches.Patch(color='red', label='A_{ij} > 0'),
        mpatches.Patch(color='orange', label='A_{ij} < 0')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

# Example usage and testing
if __name__ == "__main__":
    data_name = 'norman'
    device = 'cuda'
    
    Y = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()
    X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()
    
    #KB = RegualtoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz', neg_trn_pth=f'rules/human/{data_name}_KB_N.npz', device=device)
    #KB.closure_(T=5, closure_type='weighted')
    #
    #Y_deduction = KB.deduce(torch.tensor(X).float().to(device)).to('cpu').numpy()
    Y_deduction = np.load(f'data_anal/incons_plots/{data_name}_Y_d.npy')
    
    # Step 1: Create consistency matrix Yc
    Yc = create_consistency_matrix(Y_deduction, Y)
    print(f"\nYc shape: {Yc.shape}")
    print("Yc value counts:", np.unique(Yc, return_counts=True))
    
    # Step 2: Create adjacency matrix A
    A = create_adjacency_matrix(X, Yc)
    print(f"\nA shape: {A.shape}")
    print("A value range:", A.min(), "to", A.max())
    
    # Step 3: Plot network
    print("\nPlotting network...")
    G = plot_network(A, title="Network Visualization of Consistency Matrix A")
    
    # Additional analysis
    print(f"\nNetwork statistics:")
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of edges: {len(G.edges())}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
    
    # Show value distribution in A
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(A.flatten(), bins=30, alpha=0.7, color='skyblue')
    plt.title('Distribution of A values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    values, counts = np.unique(A, return_counts=True)
    plt.bar(values, counts, alpha=0.7, color='lightcoral')
    plt.title('Unique value counts in A')
    plt.xlabel('Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'data_anal/incons_plots/{data_name}_network.png', dpi=600)
    plt.show()
