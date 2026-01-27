import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Assuming your data is in a DataFrame called 'expr_df' (genes x samples)
# And you have a 'groups' Series with 7 group labels for the 68 samples

# Example data setup (replace with your actual data)
#expr_df = pd.DataFrame(np.random.randint(0, 100, size=(1000, 68)),
#                      columns=[f'Sample_{i}' for i in range(1, 69)],
#                      index=[f'Gene_{i}' for i in range(1, 1001)])
expr_df = pd.read_csv('dataset/counts.csv', index_col=0).transpose()
metadata_df = pd.read_csv('dataset/metadata.csv', index_col=0)
expr_df = expr_df[metadata_df.index]

groups = pd.Series(np.repeat(['ERP013795:\nmazF / mqsR',
                              'ERP109792:\nmazF / mqsR',
                              'ERP111806:\naroP / bamC /\n cusA / hemX /\n topA / yicR / yqjD',
                              'ERP119261:\nryhB / cyaR / arcZ /\n gcvB / micA', 
                              'ERP007009:\nryhB',
                              'SRP068508:\ndicF', 'SRP166315:\nrydC'], 
                             [10, 4, 14, 30, 4, 3, 3]),
                  index=expr_df.columns)

# Define expression level thresholds (adjust based on your data)

# Classify expression levels
expr_levels = pd.DataFrame({
    '< 20': (expr_df <= 20).sum(axis=0),
    '20 - 100': ((expr_df > 20) & (expr_df <= 100)).sum(axis=0),
    '100 - 1000': ((expr_df > 100) & (expr_df <= 500)).sum(axis=0),
    '1000 - 10000': ((expr_df > 500) & (expr_df <= 10000)).sum(axis=0),
    '> 10000': (expr_df > 10000).sum(axis=0)
}, index=expr_df.columns)

# Convert to proportions
proportions = expr_levels.div(expr_levels.sum(axis=1), axis=0)

# Set up polar plot with group coloring
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, polar=True)

# Create group color mapping
group_colors = plt.cm.get_cmap('tab20', 7)
group_labels = groups.unique()
group_color_dict = {g: group_colors(i) for i, g in enumerate(group_labels)}

# Get sample order and group assignments
theta = np.linspace(0, 2*np.pi, len(expr_df.columns), endpoint=False)
width = 2 * np.pi / len(expr_df.columns)

# Plot stacked bars with group coloring on edges
bottom = np.zeros(len(expr_df.columns))
for i, level in enumerate(['< 20', '20 - 100', '100 - 1000', '1000 - 10000', '> 10000']):
    ax.bar(theta, proportions[level], width=width, bottom=bottom,
           color=['#1f77b4', '#2ca02c', 'yellowgreen', '#ff7f0e', 'yellow'][i], 
           edgecolor=['lightgray'],
           linewidth=.5, alpha=0.8, label=level)
    bottom += proportions[level]

# Add group separator lines and labels
group_boundaries = [- width/2]
ax.plot([2*np.pi - width/2, 2*np.pi - width/2], [0, 1], color='black', linestyle='--', alpha=0.5)
current_group = groups[0]
for i, sample in enumerate(expr_df.columns):
    if groups[sample] != current_group:
        ax.plot([theta[i]-width/2, theta[i]-width/2], [0, 1], color='black', linestyle='--', alpha=0.5)
        group_boundaries.append(theta[i] - width/2)
        current_group = groups[sample]
group_boundaries.append(2*np.pi - width/2)

# Add group labels in the center
for i in range(len(group_boundaries)-1):
    mid_angle = (group_boundaries[i] + group_boundaries[i+1]) / 2
    ax.text(mid_angle, 1.2, group_labels[i], 
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            color='black')

# Customize plot
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title('Alignment Result of 68 Signle Gene Overexpression Data Samples', 
             pad=100, fontsize=18, loc='center')
ax.set_xticks(theta)
ax.set_xticklabels(expr_df.columns, fontsize=6, rotation=90)
ax.set_yticklabels([])
ax.grid(True, alpha=0.3)

# Create custom legend
expr_legend = [Patch(facecolor='#1f77b4', label='< 20'),
               Patch(facecolor='#2ca02c', label='20 - 100'),
               Patch(facecolor='yellowgreen', label='100 - 500'),
               Patch(facecolor='#ff7f0e', label='500 - 10000'),
               Patch(facecolor='yellow', label='> 10000')]
#group_legend = [Patch(facecolor=group_color_dict[g], label=g) for g in group_labels]

plt.legend(handles=expr_legend,# + group_legend, 
           loc='upper right', 
           bbox_to_anchor=(1.3, 1.1),
           title='Sequence Counts')

plt.tight_layout()
plt.savefig(f'data_anal/expression_plots/count_polar.png', dpi=300)
plt.show()
