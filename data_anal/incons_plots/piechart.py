import numpy as np
import torch
from scipy.sparse import load_npz
import os
import matplotlib.pyplot as plt
import seaborn as sns

from egoal.reasoner import RegualtoryKB
from egoal.learner_refl import ReflectLearner


# Donut chart
# Nested donut chart
def nested_donut_chart():
    """Create a nested donut chart"""
    # Outer ring data
    main_categories = ['Product A', 'Product B', 'Product C']
    main_values = [50, 30, 20]
    main_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    # Inner ring data (subcategories)
    sub_values = [
        [20, 15, 15],  # Subcategories for Product A
        [10, 12, 8],   # Subcategories for Product B
        [8, 7, 5]      # Subcategories for Product C
    ]
    sub_colors = ['#ff9999', '#ffcccc', '#ffe6e6']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Outer ring
    outer_wedges, outer_texts = ax.pie(main_values, 
                                      radius=1.3,
                                      colors=main_colors,
                                      startangle=90,
                                      wedgeprops=dict(width=0.3, edgecolor='w'))
    
    # Inner ring
    inner_wedges, inner_texts = ax.pie([item for sublist in sub_values for item in sublist],
                                      radius=1.3-0.3,
                                      colors=sub_colors*3,
                                      startangle=90,
                                      wedgeprops=dict(width=0.2, edgecolor='w'))
    
    ax.axis('equal')
    plt.title('Product Sales with Subcategories', fontsize=14, fontweight='bold')
    
    # Add legend
    plt.legend(outer_wedges, main_categories, title="Main Products", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()


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

n_consit = np.sum((Y == Y_deduction) & (Y != 0))
n_incomp = np.sum((Y != Y_deduction) & (Y_deduction == 0))
n_incons = np.sum((Y != Y_deduction) & (Y_deduction != 0))
n_empty = np.sum((Y == Y_deduction) & (Y_deduction == 0))
total = n_consit + n_incomp + n_incons + n_empty
n_consit, n_incomp, n_incons, n_empty = n_consit/total, n_incomp/total, n_incons/total, n_empty/total


"""Create a donut chart"""
categories = ['Consistent', 'Misssing in KB', 'Data-KB Conflict']#, 'Not Annotated']
values = [n_consit, n_incomp, n_incons]
colors = ['green', 'orange', 'red']
#explode = [0, 0, 0, 0.1]

fig, ax = plt.subplots(figsize=(8, 6))

# Create pie chart and remove the center to make it a donut
wedges, texts, autotexts = ax.pie(values, 
                                 #explode=explode,
                                 labels=categories,
                                 colors=colors,
                                 autopct='%1.1f%%',
                                 startangle=90,
                                 wedgeprops=dict(width=0.3))  # Width controls donut thickness


ax.axis('equal')
plt.title(f'Data-KB Consistency: {data_name} dataset vs {kb_name} KB', fontsize=14, fontweight='bold')
plt.savefig(f'data_anal/incons_plots/{data_name}_{kb_name}_piechart.png', dpi=600)
plt.show()

