import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example DataFrame (replace with your data)
df_lfc = pd.read_csv('dataset/logfc_gene.csv', index_col=0)
df_pv = pd.read_csv('dataset/pvalue_gene.csv', index_col=0)
df_meta = pd.read_csv('dataset/metadata.csv', index_col=0)

# Calculate -log10(p-value) for better visualization

# Define thresholds (adjust as needed)
log2FC_threshold = 1.0
p_value_threshold = 0.05

# Classify genes based on thresholds
for idx,row in df_meta.iterrows():
    if idx not in df_pv.index:
        continue

    gene = row['overexpression']
    df = pd.DataFrame({'log2FC':df_lfc.loc[idx,:], 'p_value':df_pv.loc[idx,:]})
    df['neg_log10_p'] = -np.log10(df['p_value'])
    df['Significance'] = 'Not Significant'
    df.loc[(df['log2FC'] >= log2FC_threshold) & (df['p_value'] <= p_value_threshold), 'Significance'] = 'Upregulated'
    df.loc[(df['log2FC'] <= -log2FC_threshold) & (df['p_value'] <= p_value_threshold), 'Significance'] = 'Downregulated'
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df['log2FC'],
        df['neg_log10_p'],
        c=df['Significance'].map({'Upregulated': 'red', 'Downregulated': 'blue', 'Not Significant': 'gray'}),
        alpha=0.6,
        s=30
    )
    
    # Add threshold lines
    plt.axvline(x=log2FC_threshold, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(x=-log2FC_threshold, color='black', linestyle='--', linewidth=0.8)
    plt.axhline(y=-np.log10(p_value_threshold), color='black', linestyle='--', linewidth=0.8)
    
    # Labels and title
    plt.xlabel('log2 Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.title(f'Gene Expression of Sample {idx}: {gene} Overexpression')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Upregulated', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Downregulated', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Not Significant', markerfacecolor='gray', markersize=8)
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(f'data_anal/expression_plots/{idx}_{gene}.png', dpi=300)
    #plt.show()
    plt.close()
