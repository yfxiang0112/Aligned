import numpy as np
import torch
from scipy.sparse import load_npz
import os
import matplotlib.pyplot as plt

from egoal.reasoner import RegualtoryKB
from egoal.learner_refl import ReflectLearner


def create_stacked_bar_plot(vector1, vector2, vector3, vector4,
                           colors=['green', 'orange', 'red', 'lightgray'],
                           labels=['Vector 1', 'Vector 2', 'Vector 3', 'Vector 4'],
                           figsize=(14, 8), alpha=0.8,
                           data_name='Norman', kb_name='GO'):
    """
    Create a stacked bar plot for three vectors of the same length.
    Each vector's bars are stacked on top of the previous one.
    
    Parameters:
    vector1, vector2, vector3: numpy arrays of same length
    colors: list of colors for each vector
    labels: list of labels for each vector
    figsize: figure size
    alpha: transparency level
    """
    
    # Validate input lengths
    if len(vector1) != len(vector2) or len(vector1) != len(vector3) or len(vector1) != len(vector4):
        raise ValueError("All vectors must have the same length")
    
    n = len(vector1)
    x = np.arange(n)  # x positions
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked bars
    bar_width = 1.  # Width of bars
    
    # Bottom layer: vector1 (red)
    bars1 = ax.bar(x, vector1, width=bar_width, 
                   color=colors[0], alpha=alpha, 
                   label=labels[0],  linewidth=0.5)
    
    # Middle layer: vector2 (green) stacked on vector1
    bars2 = ax.bar(x, vector2, width=bar_width, bottom=vector1,
                   color=colors[1], alpha=alpha, 
                   label=labels[1],  linewidth=0.5)
    
    # Top layer: vector3 (orange) stacked on vector1 + vector2
    bars3 = ax.bar(x, vector3, width=bar_width, bottom=vector1 + vector2,
                   color=colors[2], alpha=alpha, 
                   label=labels[2],  linewidth=0.5)
    
    # Top layer: vector3 (orange) stacked on vector1 + vector2
    bars4 = ax.bar(x, vector4, width=bar_width, bottom=vector1 + vector2 + vector3,
                   color=colors[3], alpha=alpha, 
                   label=labels[3],  linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Regulated Genes in Regulatory Networks')
    ax.set_ylabel('Portion')
    ax.set_title(f'Consistent Edges in Regulatory Network with Dataset\n{data_name} Data vs {kb_name} KB')
    ax.legend()
    
    # Adjust x-axis for better visibility with many bars
    if n > 100:
        # For large datasets, show fewer x-ticks
        ax.set_xticks(np.linspace(0, n-1, min(20, n//100)))
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_xticks(x)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    
    return fig, ax


# Example usage with sample data
def main():

    data_name = 'norman'
    kb_name = 'dorothea'
    
    device = 'cuda'
    
    Y = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()
    X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()
    
    
    if os.path.exists(f'data_anal/incons_plots/{data_name}_Y_d.npy'):
        Y_deduction = np.load(f'data_anal/incons_plots/{data_name}_Y_d.npy')
    else:
        KB = RegualtoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz', neg_trn_pth=f'rules/human/{data_name}_KB_N.npz', device=device)
        KB.closure_(T=5, closure_type='weighted')
        
        Y_deduction = KB.deduce(torch.tensor(X).float().to(device)).to('cpu').numpy()
        np.save(f'data_anal/incons_plots/{data_name}_Y_d.npy', Y_deduction)
    
    n_consit = np.sum((Y == Y_deduction) & (Y != 0), axis=0)
    n_incomp = np.sum((Y != Y_deduction) & (Y_deduction == 0), axis=0)
    n_incons = np.sum((Y != Y_deduction) & (Y_deduction != 0), axis=0)
    n_empty = np.sum((Y == Y_deduction) & (Y_deduction == 0), axis=0)
    total = n_consit + n_incomp + n_incons + n_empty
    n_consit, n_incomp, n_incons, n_empty = n_consit/total, n_incomp/total, n_incons/total, n_empty/total
    #n_consit = np.clip(n_consit - 1e-3, 0, 1.)
    #n_incomp = np.clip(n_incomp - 1e-3, 0, 1.)
    #n_incons = np.clip(n_incons - 1e-3, 0, 1.)

    sort_idx = np.argsort(n_incons)[::-1]
    n_consit, n_incomp, n_incons, n_empty = n_consit[sort_idx], n_incomp[sort_idx], n_incons[sort_idx], n_empty[sort_idx]
    sort_idx = np.argsort(n_incomp)[::-1]
    n_consit, n_incomp, n_incons, n_empty = n_consit[sort_idx], n_incomp[sort_idx], n_incons[sort_idx], n_empty[sort_idx]

    sort_idx = np.argsort(n_consit)[::-1]
    n_consit, n_incomp, n_incons, n_empty = n_consit[sort_idx], n_incomp[sort_idx], n_incons[sort_idx], n_empty[sort_idx]

    
    print(f"Vector shapes: {n_consit.shape}, {n_incomp.shape}, {n_incons.shape}")
    print(f"Vector 1 range: [{n_consit.min():.2f}, {n_consit.max():.2f}]")
    print(f"Vector 2 range: [{n_incomp.min():.2f}, {n_incomp.max():.2f}]")
    print(f"Vector 3 range: [{n_incons.min():.2f}, {n_incons.max():.2f}]")
    
    # Option 1: Full stacked bar plot (may be dense for 5000 points)
    print("\nCreating full stacked bar plot...")
    fig1, ax1 = create_stacked_bar_plot(
        n_consit, n_incomp, n_incons, n_empty,
        labels=['Consistent Edges', 'Incomplete Edges', 'Inconsistent Edges', 'Not Annotated'],
        data_name=data_name, kb_name=kb_name
    )
    
    plt.tight_layout()
    plt.savefig(f'data_anal/incons_plots/{data_name}_barplot.png', dpi=600)
    plt.show()
    
    # Option 3: For very large datasets, consider area plot
    #n = len(n_consit)
    #print("\nCreating area plot for full dataset...")
    #plt.figure(figsize=(14, 8))
    #
    #x_full = np.arange(n)
    #plt.fill_between(x_full, 0, n_consit, color='red', alpha=0.7, label='Base Layer')
    #plt.fill_between(x_full, n_consit, n_consit + n_incomp, color='green', alpha=0.7, label='Middle Layer')
    #plt.fill_between(x_full, n_consit + n_incomp, n_consit + n_incomp + n_incons, color='orange', alpha=0.7, label='Top Layer')
    #
    #plt.xlabel('Index')
    #plt.ylabel('Cumulative Values')
    #plt.title('Area Plot of Three Vectors (Stacked)')
    #plt.legend()
    #plt.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    main()
