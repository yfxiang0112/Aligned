import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import seaborn as sns

save_name = 'reconstruction'
log_path = 'results/ex2_refinement/log_mix.csv'
baseline_log_path = 'results/ex2_refinement/log_mix_baseline.csv'

gsr_log_path = 'results/ex2_refinement/gsr_mix.csv'
gsr_baseline_log_path = 'results/ex2_refinement/gsr_mix_baseline.csv'

# Set publication-quality style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],#, 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'legend.fontsize': 16,
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
df = pd.read_csv(log_path, index_col=0)
df_baseline = pd.read_csv(baseline_log_path, index_col=0)
df_gsr= pd.read_csv(gsr_log_path, index_col=['p_incomp', 'score_type'])
df_gsr_baseline= pd.read_csv(gsr_baseline_log_path, index_col=['p_incomp', 'score_type'])

x_labels = ['Original\nKB', '0%\n(Control)', '5%', '10%', '20%', '30%', '40%', '50%', '70%', '90%']
x_pos = np.arange(len(x_labels))

# Define color palette
#colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Blue, Green, Red, Purple
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Colorblind-friendly palette
markers = ['o', 's']  # Circle, Square

# =============================================================================
# Figure 1: Reconstruction F1 Scores
# =============================================================================
fig = plt.figure(figsize=(20, 6.5))
ax1 = fig.add_subplot(1, 3, 1)

y_reg = df['f1_initial_KB_combined_mean']
e_reg = df['f1_initial_KB_combined_tol']
y_clo = df['f1_closure_KB_combined_mean']
e_clo = df['f1_closure_KB_combined_tol']

b_reg = df_baseline['f1_initial_KB_combined_mean']
b_clo = df_baseline['f1_closure_KB_combined_mean']

lineh = ax1.axhline(y=y_reg[0], color='gray', linestyle='--', linewidth=.8, alpha=0.8)

# Plot lines with error bars
line1 = ax1.errorbar(x_pos, y_reg, yerr=e_reg, fmt=markers[0], color=colors[1],
             markersize=5, capsize=2.5, capthick=.8, elinewidth=.8,
             alpha=0.9, zorder=4)
ax1.plot(x_pos, y_reg, '-', color=colors[1], linewidth=1.5, alpha=0.8, zorder=3)
line3, = ax1.plot(x_pos, b_reg, '--', color=colors[1], linewidth=1.2, alpha=0.8, zorder=3)

line2 = ax1.errorbar(x_pos, y_clo, yerr=e_clo, fmt=markers[1], color=colors[0],
             markersize=5, capsize=2.5, capthick=.8, elinewidth=.8,
             alpha=0.9, zorder=4)
ax1.plot(x_pos, y_clo, '-', color=colors[0], linewidth=1.5, alpha=0.8, zorder=3)
line4, = ax1.plot(x_pos, b_clo, '--', color=colors[0], linewidth=1.2, alpha=0.8, zorder=3)

# Customize axes
ax1.set_xlabel('Noise Interactions (%)',  labelpad=-5)
ax1.set_ylabel('$F_1$ Score',  labelpad=5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels, rotation=45, ha='right')
ax1.set_title('(a) Interaction Accuracy', fontweight='bold', pad=10)

# Set y-axis limits for better visualization
y_min = min(b_reg.min(), b_clo.min()) - 0.05
y_max = max(y_reg.max(), y_clo.max()) + 0.05
ax1.set_ylim(y_min, y_max)

# Add grid and clean spines
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add legend
ax1.legend(handles = [line1, line2, (line3,line4), lineh],
           labels = ['Direct Interac.', 'Indirect Interac.',
                     'Non-sparse', 'Original GRN'],
           handler_map = {tuple: HandlerTuple(ndivide=None)},
           loc='lower left', frameon=True, framealpha=1.0, edgecolor='black')

# =============================================================================
# Figure 2: Structural Scores
# =============================================================================
ax2 = fig.add_subplot(1, 3, 2)

y_mod = df['modularity_mean']
e_mod = df['modularity_tol']
y_aso = df['degree_assortativity_mean']
e_aso = df['degree_assortativity_tol']

b_mod = df_baseline['modularity_mean']
b_aso = df_baseline['degree_assortativity_mean']

lineh = ax2.axhline(y=y_aso[0], color='gray', linestyle='--', linewidth=.8, alpha=0.8)
lineh = ax2.axhline(y=y_mod[0], color='gray', linestyle='--', linewidth=.8, alpha=0.8)

# Plot lines with error bars
line1 = ax2.errorbar(x_pos, y_mod, yerr=e_mod, fmt=markers[0], color=colors[2],
             markersize=5, capsize=2.5, capthick=.8, elinewidth=.8,
              alpha=0.9, zorder=4)
ax2.plot(x_pos, y_mod, '-', color=colors[2], linewidth=1.5, alpha=0.8, zorder=3)
line3, = ax2.plot(x_pos, b_mod, '--', color=colors[2], linewidth=1.2, alpha=0.8, zorder=3)

line2 = ax2.errorbar(x_pos, y_aso, yerr=e_aso, fmt=markers[1], color=colors[3],
             markersize=5, capsize=2.5, capthick=.8, elinewidth=.8,
              alpha=0.9, zorder=4)
ax2.plot(x_pos, y_aso, '-', color=colors[3], linewidth=1.5, alpha=0.8, zorder=3)
line4, = ax2.plot(x_pos, b_aso, '--', color=colors[3], linewidth=1.2, alpha=0.8, zorder=3)

ax2.set_ylim([-.67,.45])

# Customize axes
ax2.set_xlabel('Noise Interactions (%)', labelpad=-5)
ax2.set_ylabel('Topological Score', labelpad=5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels, rotation=45, ha='right')
ax2.set_title('(b) Network Topology',
               fontweight='bold', pad=10)

# Add grid and clean spines
ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add legend
ax2.legend(handles = [line1, line2, (line3,line4), lineh],
           labels = ['Modularity ($\\uparrow$)', 'Assortativity',
                     'Non-sparse', 'Original GRN'],
           handler_map = {tuple: HandlerTuple(ndivide=None)},
           loc='lower left', frameon=True, framealpha=1.0, edgecolor='black')


# =============================================================================
# Figure 3: Biological Meaningfulness
# =============================================================================
ctrl_row = np.array(df_gsr.loc[('orig','mean_auprc_w')])
pathways = df_gsr.loc[[x for x in df_gsr.index\
        if x[1]=='mean_auprc_w' and x[0] in ['0.05','0.1','0.3','0.5']]] - ctrl_row
pathways_baseline = df_gsr_baseline.loc[[x for x in df_gsr_baseline.index\
        if x[1]=='mean_auprc_w' and x[0] in ['0.05','0.1','0.3','0.5']]] - ctrl_row

x_labels = ['5%', '10%', '30%', '50%']

pathways['x_labels'] = x_labels
pathways = pathways.reset_index(drop=True).set_index('x_labels', drop=True)

pathways_baseline['x_labels'] = [x+'\nBaseline' for x in x_labels]
pathways_baseline = pathways_baseline.reset_index(drop=True).set_index('x_labels', drop=True)

ax3 = fig.add_subplot(1, 3, 3)

data_for_plot = []
for vec_index, vec_name in enumerate(pathways_baseline.index):
    for dim_value in pathways_baseline.loc[vec_name]:
        data_for_plot.append({'Removed Arcs': vec_name, 'Deviation': dim_value, 'Group': 'Non-sparse'})

for vec_index, vec_name in enumerate(pathways.index):
    for dim_value in pathways.loc[vec_name]:
        data_for_plot.append({'Removed Arcs': vec_name, 'Deviation': dim_value, 'Group': 'ALIGNED'})

data_for_plot = pd.DataFrame(data_for_plot)
print(data_for_plot)

#sns.violinplot(data=data_for_plot, x='Removed Arcs', y='Deviation', ax=ax3, cut=0, inner='box') 
sns.violinplot(data=data_for_plot, x='Removed Arcs', y='Deviation', hue='Group',
               ax=ax3, cut=0, inner='box', dodge=False,
               palette={'Non-sparse': '#1f77b4',
                        'ALIGNED': '#ff7f0e'})  # Distinct colors


ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.8, label='Original GRN')
ax3.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

# Improve labels and title 
ax3.set_xlabel('Noise Interactions (%)', labelpad = 25)
ax3.set_ylabel('Deviation from Original KB')
ax3.set_title('(c) Pathway Enrichment Scores', fontweight='bold', pad=10)

# Improve tick labels for readability
ax3.tick_params(axis='both', which='major')
#plt.setp(ax3.get_xticklabels(), rotation=45, ha='right') # Rotate labels if long
ax3.set_xticklabels(x_labels*2, rotation=45, ha='right')
ax3.grid(axis='y', linestyle=':', alpha=0.4)

# Optional: Add a legend for the reference line
ax3.legend(loc='lower right', frameon=True)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

group_x_positions = [1.5, 5.5]  # Middle of each group
group_labels = ['Non-Sparse Baseline', 'Knowledge Refinement']
for x_pos, label in zip(group_x_positions, group_labels):
    ax3.text(x_pos, -0.12, label, ha='center', va='top', 
            transform=ax3.get_xaxis_transform())


# Adjust layout and save
plt.tight_layout()
plt.savefig(f'results/fig5_refine/{save_name}.png', bbox_inches='tight', pad_inches=0.05)
plt.savefig(f'results/fig5_refine/{save_name}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig(f'results/fig5_refine/{save_name}.pgf', format='pgf', dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()
