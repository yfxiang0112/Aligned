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
df = pd.read_csv('data_anal/refine/log.csv', index_col=0)
x_labels = ['Original\nKB', 'Baseline\n0%', '5%', '10%', '20%', '30%', '40%', '50%', '70%', '90%']
x_pos = np.arange(len(x_labels))

# Define color palette
#colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Blue, Green, Red, Purple
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Colorblind-friendly palette
markers = ['o', 's']  # Circle, Square

# =============================================================================
# Figure 1: Structural Scores
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(5, 4))  # Single column width

y_mod = df['modularity_mean']
e_mod = df['modularity_tol']
y_aso = df['degree_assortativity_mean']
e_aso = df['degree_assortativity_tol']

# Plot lines with error bars
ax1.errorbar(x_pos, y_mod, yerr=e_mod, fmt=markers[0], color=colors[0],
             markersize=5, capsize=2.5, capthick=1.2, elinewidth=1.2,
             label='Modularity', alpha=0.9, zorder=4)
ax1.plot(x_pos, y_mod, '-', color=colors[0], linewidth=1.5, alpha=0.8, zorder=3)

ax1.errorbar(x_pos, y_aso, yerr=e_aso, fmt=markers[1], color=colors[1],
             markersize=5, capsize=2.5, capthick=1.2, elinewidth=1.2,
             label='Degree assortativity', alpha=0.9, zorder=4)
ax1.plot(x_pos, y_aso, '-', color=colors[1], linewidth=1.5, alpha=0.8, zorder=3)

# Customize axes
ax1.set_xlabel('Removed edges (%)', fontsize=11, labelpad=5)
ax1.set_ylabel('Structural score', fontsize=11, labelpad=5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels, rotation=45, ha='right')
ax1.set_title('Structural properties of reconstructed networks', fontsize=12, pad=10)

# Add grid and clean spines
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add legend
ax1.legend(loc='lower left', frameon=True, framealpha=1.0, edgecolor='black')

# Adjust layout and save
plt.tight_layout()
plt.savefig('data_anal/refine/structural.png', bbox_inches='tight', pad_inches=0.05)
plt.savefig('data_anal/refine/structural.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('data_anal/refine/structural.pgf', format='pgf', dpi=600, bbox_inches='tight', pad_inches=0.05)

# =============================================================================
# Figure 2: Reconstruction F1 Scores
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(5, 4))  # Single column width

y_reg = df['f1_initial_KB_combined_mean']
e_reg = df['f1_initial_KB_combined_tol']
y_clo = df['f1_closure_KB_combined_mean']
e_clo = df['f1_closure_KB_combined_tol']

# Plot lines with error bars
ax2.errorbar(x_pos, y_reg, yerr=e_reg, fmt=markers[0], color=colors[0],
             markersize=5, capsize=2.5, capthick=1.2, elinewidth=1.2,
             label='Reconstructed GRN', alpha=0.9, zorder=4)
ax2.plot(x_pos, y_reg, '-', color=colors[0], linewidth=1.5, alpha=0.8, zorder=3)

ax2.errorbar(x_pos, y_clo, yerr=e_clo, fmt=markers[1], color=colors[2],
             markersize=5, capsize=2.5, capthick=1.2, elinewidth=1.2,
             label=r'Reconstructed $R^{(k)}$', alpha=0.9, zorder=4)
ax2.plot(x_pos, y_clo, '-', color=colors[2], linewidth=1.5, alpha=0.8, zorder=3)

# Customize axes
ax2.set_xlabel('Removed edges (%)', fontsize=11, labelpad=5)
ax2.set_ylabel('F1 score', fontsize=11, labelpad=5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels, rotation=45, ha='right')
ax2.set_title('Reconstruction accuracy', fontsize=12, pad=10)

# Set y-axis limits for better visualization
y_min = min(y_reg.min(), y_clo.min()) - 0.05
y_max = max(y_reg.max(), y_clo.max()) + 0.05
ax2.set_ylim(y_min, y_max)

# Add grid and clean spines
ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add legend
ax2.legend(loc='lower left', frameon=True, framealpha=1.0, edgecolor='black')

# Adjust layout and save
plt.tight_layout()
plt.savefig('data_anal/refine/reconstruction_f1.png', bbox_inches='tight', pad_inches=0.05)
plt.savefig('data_anal/refine/reconstruction_f1.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('data_anal/refine/reconstruction_f1.pgf', format='pgf', dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()
