import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication-quality style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],#, 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 600,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

# Load data

df = pd.read_csv('data_anal/experiment_results/results.csv', index_col=0)
x_labels = ['Data only', 'Integration 1', 'Refinement 1', 'Integration 2', 'Refinement 2']
x_pos = np.arange(len(x_labels))

datasets = ['norman', 'dixit', 'adamson']


# Define color palette
#colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Blue, Green, Red, Purple
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Colorblind-friendly palette
markers = ['o', 's']  # Circle, Square

# =============================================================================
# Figure 1: Structural Scores
# =============================================================================
fig = plt.figure(figsize=(20, 5.5))
fig.patch.set_facecolor('white')

n_datasets = len(datasets)
# Create subplot for each dataset with improved spacing
for i, dataset in enumerate(datasets):
    data = df[(df['data_name']==dataset) & (df['score']=='integrated') & (df['stage']!='init')] 
    data_GNN = data[data['model']=='GNN']
    data_MLP = data[data['model']=='MLP']

    ax = fig.add_subplot(1, n_datasets, i+1)

    ax.errorbar(x_pos, data_GNN['data_f1_mean'], yerr=data_GNN['data_f1_stde'],
                fmt=markers[0], color=colors[0],
                markersize=5, capsize=2.5, capthick=1.2, elinewidth=1.2,
                label='Data Consistency' if i==0 else None, alpha=0.9, zorder=4)
    ax.plot(x_pos, data_GNN['data_f1_mean'], '-', color=colors[0], linewidth=1.5, alpha=0.8, zorder=3)
    
    ax.errorbar(x_pos, data_GNN['kb_f1_mean'], yerr=data_GNN['kb_f1_stde'],
                fmt=markers[1], color=colors[1],
                markersize=5, capsize=2.5, capthick=1.2, elinewidth=1.2,
                label='Knowledge Consistency' if i==0 else None, alpha=0.9, zorder=4)
    ax.plot(x_pos, data_GNN['kb_f1_mean'], '-', color=colors[1], linewidth=1.5, alpha=0.8, zorder=3)
    
    # Customize axes
    ax.set_xlabel('ABL Stage', fontsize=11, labelpad=5)
    ax.set_ylabel('F_1 score', fontsize=11, labelpad=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_title(f'{dataset.capitalize()} et al. Dataset', fontsize=12, pad=10)
    
    # Add grid and clean spines
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
# Add legend
fig.legend(frameon=True, framealpha=1.0, edgecolor='black')

# Adjust layout and save
plt.tight_layout()
plt.savefig('data_anal/experiment_results/line_plots.png', bbox_inches='tight', pad_inches=0.05)
plt.savefig('data_anal/experiment_results/line_plots.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('data_anal/experiment_results/line_plots.pgf', format='pgf', dpi=600, bbox_inches='tight', pad_inches=0.05)

