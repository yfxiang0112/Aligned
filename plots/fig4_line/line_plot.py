import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication-quality style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],#, 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
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
x_labels = ['Baseline', 'Integration 1', 'Refinement 1', 'Integration 2', 'Refinement 2']
x_pos = np.arange(len(x_labels))

datasets = ['norman', 'dixit', 'adamson']


# Define color palette
#colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Blue, Green, Red, Purple
#colors = ['#2E86AB', '#2ca02c', '#A23B72', '#F18F01', '#C73E1D']  # Colorblind-friendly palette
#colors = ['#4E79A7', '#59A14F', '#B07AA1']
colors = ['#2E86AB', '#59A14F', '#A23B72']
markers = ['o']  # Circle, Square

fig = plt.figure(figsize=(19, 5))
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
                markersize=6, capsize=2.5, capthick=.8, elinewidth=.8,
                label='Data Consistency' if i==0 else None, alpha=0.9, zorder=4)
    ax.plot(x_pos, data_GNN['data_f1_mean'], '-', color=colors[0], linewidth=2., alpha=0.8, zorder=3)
    
    ax.errorbar(x_pos, data_GNN['kb_f1_mean'], yerr=data_GNN['kb_f1_stde'],
                fmt=markers[0], color=colors[1],
                markersize=6, capsize=2.5, capthick=.8, elinewidth=.8,
                label='Knowledge Consistency' if i==0 else None, alpha=0.9, zorder=4)
    ax.plot(x_pos, data_GNN['kb_f1_mean'], '-', color=colors[1], linewidth=2., alpha=0.8, zorder=3)

    ax.errorbar(x_pos, data_GNN['bal_f1_mean'], yerr=data_GNN['bal_f1_stde'],
                fmt=markers[0], color=colors[2],
                markersize=6, capsize=2.5, capthick=.8, elinewidth=.8,
                label='Balanced Consistency' if i==0 else None, alpha=0.9, zorder=4)
    ax.plot(x_pos, data_GNN['bal_f1_mean'], '-', color=colors[2], linewidth=2., alpha=0.8, zorder=3)
    
    # Customize axes
    #ax.set_xlabel('ABL Stage', fontsize=18, labelpad=5)
    ax.set_ylabel('$F_1$ Score',  labelpad=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=15, ha='right',  fontweight='bold')
    ax.set_title(f'({["a","b","c"][i]}) {dataset.capitalize()} et al. Dataset',  fontweight='bold', pad=10)
    
    # Add grid and clean spines
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
# Add legend
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, wspace=0.4)
#fig.legend(frameon=True, framealpha=1.0, fontsize=18, edgecolor='black')
#handles, labels = ax.get_legend_handles_labels()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.03),
           ncol=3,  frameon=True, fancybox=True, shadow=True,
           facecolor='white', edgecolor='gray')


# Adjust layout and save
#plt.tight_layout()
plt.savefig('plots/fig4_line/line_plots.png', bbox_inches='tight', pad_inches=0.05)
plt.savefig('plots/fig4_line/line_plots.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('plots/fig4_line/line_plots.pgf', format='pgf', dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()
