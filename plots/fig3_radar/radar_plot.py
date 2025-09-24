import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    "pgf.texsystem": "pdflatex",
    #"pgf.preamble": r"\\usepackage[utf8]{inputenc}\\usepackage[T1]{fontenc}\\usepackage{fontspec}",
})
# Set publication-quality style
#plt.style.use('default')
#plt.rcParams.update({
#    'font.family': 'serif',
#    'font.serif': ['Times New Roman'],#, 'DejaVu Serif'],
#    'font.size': 11,
#    'axes.labelsize': 11,
#    'axes.titlesize': 12,
#    'legend.fontsize': 9,
#    'xtick.labelsize': 9,
#    'ytick.labelsize': 9,
#    'figure.dpi': 600,
#    'lines.linewidth': 1.2,
#    'lines.markersize': 6,
#    'errorbar.capsize': 3,
#    'axes.linewidth': 0.8,
#    'grid.linewidth': 0.5,
#    'grid.alpha': 0.3,
#})

def create_radar_plot(ax, dataset_name, algorithm_values, algorithm_names, ax_num, category_labels = ['Balanced Cons.', 'Data Cons.', 'Knowledge Cons.']):
    """Create a publication-quality radar plot for one dataset comparing all algorithms"""
    # Define colors for each algorithm
    colors = {
        'ALIGNED (GNN)': '#2E86AB',         
        'ALIGNED (MLP)': '#A23B72',         
        'scFoundation': '#F18F01',
        'scGPT': '#C73E1D',       
        'GEARS': '#2D5016',       
        'Linear': "#3ECE51"
    }
    
    # Define markers for each algorithm
    markers = {
        'ALIGNED (GNN)': 'o',
        'ALIGNED (MLP)': 's', 
        'scFoundation': '^',
        'scGPT': 'D',
        'GEARS': 'v',
        'Linear': 'P'
    }
    
    # Number of variables
    N = len(category_labels)
    
    # Set specific angles: 90° (top), 210° (left), 330° (right) (converted to radians)
    # Order: Integrated F1 (top), F1 Test (left), F1 KB (right)
    angles = [np.pi/2, 7*np.pi/6, 11*np.pi/6]  # 90°, 210°, 330° in radians
    angles += angles[:1]  # Complete the circle

    max_value = max([max(x) for x in algorithm_values])
    min_value = min([min(x) for x in algorithm_values])
    max_value = round(max_value*20 + .5)/20
    min_value = round(min_value*20 - .5)/20
    print(max_value, min_value)
    
    # Plot each algorithm
    for i, (alg_name, values) in enumerate(zip(algorithm_names[::-1], algorithm_values[::-1])):
        if len(values) == 3 and all(pd.notna(values)):  # Only plot if we have all 3 values
            # Reorder values to match new angle positions: [Integrated F1, F1 Test, F1 KB]
            plot_values = [values[2], values[0], values[1]]  # integrated_f1, f1_test, f1_kb
            plot_values += plot_values[:1]  # Complete the circle
            
            # Plot algorithm with enhanced styling
            ax.plot(angles, plot_values, markers[alg_name] + '-', linewidth=2.5, label=alg_name,
                    color=colors[alg_name], alpha=0.9, markersize=6, 
                    markerfacecolor=colors[alg_name], markeredgecolor='white', markeredgewidth=1)
            ax.fill(angles, plot_values, alpha=0.1, color=colors[alg_name])
    
    # Enhanced category labels in new order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_labels, fontsize=18, fontweight='bold')
    
    # Set y-axis limits and labels with better formatting
    ax.set_ylim(min_value, max_value)
    ticks = np.linspace(start=min_value, stop=max_value, num=(int(max_value*20)-int(min_value*20)+1))\
            if max_value-min_value <= .3 else\
            np.linspace(start=round(min_value*10+.49)/10, stop=round(max_value*10-.49)/10, num=(round(max_value*10-.49)-round(min_value*10+.49)+1))
    print(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(round(x*20)/20) for x in ticks], 
                       fontsize=15, color='gray')
    
    # Enhanced grid
    ax.grid(True, alpha=0.4, linewidth=0.8, linestyle='--')
    ax.set_facecolor('#FAFAFA')  # Light background
    
    # Professional title
    ax.set_title(f'({ax_num}) {dataset_name.capitalize()} et al. Dataset', size=22, fontweight='bold', 
                pad=25, color='#2C3E50')




if __name__ == '__main__':
    datasets = ['norman', 'dixit', 'adamson']
    algorithms = ['ALIGNED (GNN)', 'ALIGNED (MLP)', 'Linear', 'GEARS', 'scGPT', 'scFoundation']

    columns = ['data_f1_mean', 'kb_f1_mean', 'bal_f1_mean']
    df_benchmk = pd.read_csv('data_anal/pert_benchmark/benchmark_results.csv')
    df_experim = pd.read_csv('data_anal/experiment_results/results.csv', index_col=0)

    plot_alg_names = {'GNN':'ALIGNED (GNN)', 'MLP':'ALIGNED (MLP)'}
    df_experim['model'] = df_experim['model'].apply(lambda x: plot_alg_names[x])
    df_experim = df_experim[(df_experim['stage']=='ABL1_refl') &\
            (df_experim['score']=='integrated') &\
            (df_experim['data_name'].isin(datasets))].set_index(['data_name', 'model'], drop=True)

    plot_alg_names = {'additive':'Linear', 'gears':'GEARS', 'scgpt':'scGPT', 'scfoundation':'scFoundation'}
    df_benchmk['model'] = df_benchmk['model'].apply(lambda x: plot_alg_names[x])
    df_benchmk.set_index(['data_name', 'model'], drop=True, inplace=True)

    df_benchmk = df_benchmk[columns]
    df_experim = df_experim[columns]
    df = pd.concat([df_benchmk, df_experim], axis=0)#.reset_index(drop=True)
    print(df)
    
    
    # Create publication-quality radar plots
    fig = plt.figure(figsize=(20, 5.5))
    fig.patch.set_facecolor('white')
    
    n_datasets = len(datasets)
    # Create subplot for each dataset with improved spacing
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(1, n_datasets, i+1, projection='polar')
        
        # Collect algorithm values and names for this dataset
        algorithm_values = []
        algorithm_names = []
        
        for algorithm in algorithms:
            #if algorithm in f1_test_pivot.columns:
            #    f1_test_val = f1_test_pivot.loc[dataset, algorithm] if algorithm in f1_test_pivot.columns else np.nan
            #    f1_kb_val = f1_kb_pivot.loc[dataset, algorithm] if algorithm in f1_kb_pivot.columns else np.nan
            #    integrated_f1_val = integrated_f1_pivot.loc[dataset, algorithm] if algorithm in integrated_f1_pivot.columns else np.nan

            f1_test_val = df.loc[(dataset, algorithm), 'data_f1_mean']
            f1_kb_val = df.loc[(dataset, algorithm), 'kb_f1_mean']
            integrated_f1_val = df.loc[(dataset, algorithm), 'bal_f1_mean']
                
            # Only include algorithms that have all three metrics
            if pd.notna(f1_test_val) and pd.notna(f1_kb_val) and pd.notna(integrated_f1_val):
                algorithm_values.append([f1_test_val, f1_kb_val, integrated_f1_val])
                algorithm_names.append(algorithm)
        
        if algorithm_values:  # Only create plot if we have data
            create_radar_plot(ax, dataset, algorithm_values, algorithm_names, ax_num=['a','b','c'][i])

    # Add a single legend for all subplots if we have any plots
    if algorithm_values:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 0.05), 
                   ncol=len(algorithm_names), fontsize=18, frameon=True, fancybox=True, shadow=True,
                   facecolor='white', edgecolor='gray')

    # Adjust layout with proper spacing
    plt.subplots_adjust(top=0.85, bottom=0.08, left=0.05, right=0.95, wspace=0.1)
    plt.show()

    # Save high-resolution version for publication
    fig.savefig(f'plots/fig3_radar/radar_plots.pdf', dpi=600, bbox_inches='tight', 
                format='pdf', facecolor='white', edgecolor='none')
    fig.savefig(f'plots/fig3_radar/radar_plots.pgf', dpi=600, bbox_inches='tight', 
                format='pgf', facecolor='white', edgecolor='none')
    fig.savefig(f'plots/fig3_radar/radar_plots.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
