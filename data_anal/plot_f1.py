import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_file = 'cross_validation_4.24'
df = pd.read_csv(f'data_anal/logs/log_{log_file}.csv', index_col=0)

# Create the bar plot
plt.figure(figsize=(12, 6))  # Adjust figure size

# Set the positions and width for the bars
x = np.arange(len(df))  # the label locations
width = 0.15  # the width of the bars

f1_type = 'macro'
# Plot each column with different colors
bars1 = plt.bar(x - 1.5*width, df[f'init_{f1_type}_f1'], width, label='random', color='lightgray')
bars2 = plt.bar(x - 0.5*width, df[f'pretrain_{f1_type}_f1'], width, label='Before ABL', color='skyblue')
bars3 = plt.bar(x + 0.5*width, df[f'abl1_{f1_type}_f1'], width, label='ABL Loop 1', color='salmon')
bars4 = plt.bar(x + 1.5*width, df[f'abl3_{f1_type}_f1'], width, label='ABL Loop 2', color='lightgreen')

# Add some text for labels, title and custom x-axis tick labels
plt.xlabel('overexpressed gene')
plt.ylabel('f1 score')
plt.title(f'{f1_type} F1 Score under Cross Validation', fontsize=14)
plt.xticks(x, df['gene_name'])  # Use row indices as x-tick labels
plt.legend()

# Optional: Add value labels on top of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

plt.tight_layout()  # Adjust layout to prevent label clipping
plt.savefig(f'data_anal/perf_plots/f1_{log_file}_{f1_type}.png', dpi=300)
plt.show()
