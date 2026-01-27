import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


df_fba = pd.read_csv('data_anal/incons_fba.tsv', sep='\t')
x_t = df_fba['test_growth'][df_fba['locus']!='WT']
x_p = df_fba['pred_growth'][df_fba['locus']!='WT']
y = df_fba['ground_growth'][df_fba['locus']!='WT']

for s,x in [('test',x_t)]:#, ('pred',x_p)]:
    plt.figure(figsize=(8, 6))
    
    # Get counts for legend
    count_zero = (y==1.).sum()
    mask_cons = (y!=1) & (abs(x-y)<.3)
    mask_incons = (y!=1) & (abs(x-y)>=.3)
    count_cons = mask_cons.sum()
    count_incons = mask_incons.sum()

    print(x[y!=1.])
    print(y[y!=1.])
    corr = x[y!=1.].corr(y[y!=1.])
    #print(x[y!=1.], y[y!=1.])
    print(corr)
    
    plt.scatter(x[mask_cons], y[mask_cons], 
                c='blue', label=f'Consistent Points (n={count_cons})', 
                alpha=0.7, edgecolors='w')
    
    # Plot points with series2≠0 in blue
    plt.scatter(x[mask_incons], y[mask_incons], 
                c='red', label=f'Inconsistent Points (n={count_incons})', 
                alpha=0.7, edgecolors='w')
    
    # Plot points with series2=0 in red
    plt.scatter(x[y==1.], [-.1]*count_zero, 
                c='gray', label=f'Points with No Growth Rate Data (n={count_zero})', 
                alpha=0.5, edgecolors='w')
    
    # Add reference line
    plt.plot([0, y[y!=1.].max()],
             [0, y[y!=1.].max()],
             'k--', label='Perfect agreement')
    
    # Customize plot
    plt.title('Scatter Plot: FBA generated growth rate vs ground truth growth rate')
    plt.xlabel('FBA Generated Growth Rate')
    plt.ylabel('Ground Truth Growth Rate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'data_anal/incons_fba_{s}.png', dpi=300)
    plt.show()
    
