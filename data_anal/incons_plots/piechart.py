import numpy as np
import torch
from scipy.sparse import load_npz, save_npz, coo_matrix
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
import json
import gc
import pandas as pd

from egoal.reasoner import RegulatoryKB
from egoal.learner_refl import ReflectLearner


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


combs = [('norman', 'omnipath'), ('norman', 'go'),
         ('dixit', 'omnipath'), ('dixit', 'go'),
         ('adamson', 'omnipath'), ('adamson', 'go'),
         ('precise1k', 'ecocyc'), ('ncbi-sra', 'ecocyc')]
device = 'cuda'

incons_dict_path = 'data_anal/incons_plots/incons_edges.json'
if not os.path.exists(incons_dict_path):
    incons_dict= {}
    for data_name, kb_name in combs:
        print(f'processing {data_name} data + {kb_name} kb')

        if os.path.exists(f'data_anal/incons_plots/data/{data_name}_Corr_P.npy')\
            and os.path.exists(f'data_anal/incons_plots/data/{data_name}_Corr_N.npy'):
            corr_P = torch.tensor(load_npz(f'data_anal/incons_plots/data/{data_name}_Corr_P.npz').toarray()).to(device)
            corr_N = torch.tensor(load_npz(f'data_anal/incons_plots/data/{data_name}_Corr_N.npz').toarray()).to(device)
        else:
            if data_name not in ['precise1k','ncbi-sra']:
                Y = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()
                X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()
            else:
                label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
                Y = np.load(f'dataset/{data_name}/Y_label.npy')
                X = np.load(f'dataset/{data_name}/X_label.npy')

            X,Y = torch.tensor(X).to(device).float(), torch.tensor(Y).to(device).float()

            
            if kb_name == 'go':
                Y = torch.abs(Y)

            unique_X, inverse_indices = torch.unique(X, dim=0, return_inverse=True)
            Y_means_P = torch.zeros((len(unique_X), Y.shape[1])).to(device)
            Y_means_N = torch.zeros((len(unique_X), Y.shape[1])).to(device)
            for i in range(len(unique_X)):
                mask = (inverse_indices == i)
                Y_means_P[i] = torch.mean(torch.clamp(Y,min=0)[mask], dim=0)
                Y_means_N[i] = torch.mean(torch.clamp(-Y,min=0)[mask], dim=0)
            
            corr_P = (torch.clamp(unique_X,min=0).T @ Y_means_P)\
                    + (torch.clamp(-unique_X,min=0).T @ Y_means_N)
            corr_N = (torch.clamp(unique_X,min=0).T @ Y_means_N)\
                    + (torch.clamp(-unique_X,min=0).T @ Y_means_P)

            save_npz(f'data_anal/incons_plots/data/{data_name}_Corr_P.npz', coo_matrix(corr_P.cpu().numpy()))
            save_npz(f'data_anal/incons_plots/data/{data_name}_Corr_N.npz', coo_matrix(corr_N.cpu().numpy()))

        
        KB = RegulatoryKB(pos_trn_pth=f'data_anal/incons_plots/data/{data_name}_{kb_name}_KB_P.npz',\
                neg_trn_pth=f'data_anal/incons_plots/data/{data_name}_{kb_name}_KB_N.npz'\
                if kb_name!='go' else None, device=device)
        KB.closure_(T=5, closure_type='weighted' if kb_name!='go' else 'naive')
        KB_true = KB.KB#.cpu().numpy()


        if data_name in ['precise1k']:
            label_set = pd.read_csv('dataset/gene_idx.csv', index_col=0)
            label_idx = np.array(label_set['precise1k_idx']!=-1)
            KB_true = KB_true[:,label_idx]

        row_idx = torch.sum(corr_P+corr_N, axis=1)>0
        corr_P, corr_N, KB_true = corr_P[row_idx], corr_N[row_idx], KB_true[row_idx]
        
        n_consit = int(torch.sum(corr_P[KB_true>0]) + torch.sum(corr_N[KB_true<0]))
        n_incomp = int(torch.sum((corr_P+corr_N)[KB_true==0]))
        n_incons = int(torch.sum((1-corr_P)[KB_true>0]) + torch.sum((1-corr_N)[KB_true<0]))
        n_empty = int(torch.sum((1-corr_P-corr_N)[KB_true==0]))
        total = n_consit + n_incomp + n_incons + n_empty
        n_consit, n_incomp, n_incons, n_empty = n_consit/total, n_incomp/total, n_incons/total, n_empty/total

        res = {'consistent': n_consit, 'missing': n_incomp, 'conflict': n_incons, 'empty': n_empty}
        incons_dict[str((data_name, kb_name))] = res

        del KB
        gc.collect()
        torch.cuda.empty_cache()

    json.dump(incons_dict, open(incons_dict_path, 'w'), indent=4)

else:
    incons_dict = json.load(open(incons_dict_path,'r'))


for k, v in incons_dict.items():
    data_name, kb_name = eval(k)[0], eval(k)[1]
    n_consit = v['consistent']
    n_incomp = v['missing']
    n_incons = v['conflict']

    green_palette = sns.color_palette("Greens", n_colors=3)  # Get 3 shades of green
    warm_palette = sns.color_palette("YlOrRd", n_colors=7)  # One less for the highlight
    
    
    """Create a donut chart"""
    categories = ['Consistent', 'Misssing\nin KB', 'Data-KB\nConflict']#, 'Not Annotated']
    values = [n_consit, n_incomp, n_incons]
    colors = [green_palette[0], warm_palette[1], warm_palette[3]]
    #explode = [0, 0, 0, 0.1]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    #ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie is circular.
    
    ax.pie([n_consit, n_incomp+n_incons],
           colors=[green_palette[0], warm_palette[5]],
           radius=0.9,
           startangle=90,
           wedgeprops=dict(width=0.3, edgecolor='white', linewidth=.5, alpha=.9),)
    
    ax.pie(values,
           #labels=categories,
           colors=colors, 
           radius=1.2,
           startangle=90,
           wedgeprops=dict(width=0.3, edgecolor='white', linewidth=.5, alpha=.85),
           autopct='%1.2f%%',
           pctdistance=0.85,
           #textprops={'fontsize': 15, 'fontweight': 'bold'})
           textprops={'fontsize': 20})
    
    # Add center circle and text
    centre_circle = plt.Circle((0, 0), 0.4, color='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{kb_name.capitalize()} KB vs\n{data_name.capitalize()} Data', ha='center', va='center', fontsize=20, fontweight='bold')

    legend_elements = [
            mpatches.Patch(facecolor=green_palette[0], label='Consistent'),
            mpatches.Patch(facecolor=warm_palette[5], label='Inconsistent'),
            mpatches.Patch(facecolor=warm_palette[1], label='  Missing in KB (FP for data)'),
            mpatches.Patch(facecolor=warm_palette[3], label='  Data-KB Conflict (FN for data)'),
    ]
    plt.legend(handles=legend_elements,
               fontsize=20,
               loc='lower right')
    
    
    ## Create pie chart and remove the center to make it a donut
    #wedges, texts, autotexts = ax.pie(values, 
    #                                 #explode=explode,
    #                                 labels=categories,
    #                                 colors=colors,
    #                                 autopct='%1.1f%%',
    #                                 startangle=90,
    #                                 wedgeprops=dict(width=0.3))  # Width controls donut thickness
    #
    #
    #ax.axis('equal')
    plt.suptitle(f'(a) Inconsistent interactions in\n {kb_name.capitalize()} KB vs {data_name.capitalize()} dataset', fontsize=24, fontweight='bold', y=.12)
    #plt.savefig(f'data_anal/incons_plots/piechart_{data_name}_{kb_name}.pgf', dpi=600, format='pgf')
    #plt.savefig(f'data_anal/incons_plots/piechart_{data_name}_{kb_name}.png', dpi=600)
    plt.show()
