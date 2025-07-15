import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def diff_matrix(true_matrix, predict_matrix, row_idx=None,col_idx=None):
    diff = np.where(
                (true_matrix[:,:]==0) & (predict_matrix[:,:]==0), 0,
            np.where(
                (true_matrix[:,:]!=0) & (predict_matrix[:,:]==0), 1,
            np.where(
                (true_matrix[:,:]==0) & (predict_matrix[:,:]!=0), 2,
            np.where(
                (true_matrix[:,:]!=0) & (predict_matrix[:,:]==true_matrix), 3, 4))))
    if row_idx:
        diff = diff[row_idx,:]
    if col_idx:
        diff = diff[:,col_idx]
    return diff

' load test labels & TRN closure results '
test_idx = [37,38,39,40,41,42,43,44,45,46,47,48, 49,50,51,52,53,54, 55,56,57, 28,29,30,58,59,60,61]
label_set = pd.read_csv('dataset/label_set_iml.csv', index_col=0)
#Y_test = np.load('dataset/ncbi-sra/Y_label.npy')[test_idx][:,list(label_set['matrix_idx'])]
Y_test = np.load('dataset/precise1k/Y_label.npy')[:,list(label_set['precise1k_idx'])]
Y_deduction = np.load('data_anal/abduction_results/Yd_ABL0.npy')
Y_pseudo = np.load('data_anal/abduction_results/Yp_ABL0.npy')

diff_deduction = diff_matrix(Y_test, Y_deduction)
diff_pseudo = diff_matrix(Y_test, Y_pseudo)


' init color mapping '
#cmap_colors = [
#    'white',    # -1
#    'lightgray',      # 0
#    'yellowgreen',      # 1
#    'lightblue',      # 2
#    'teal',      # 3
#    'cyan', # 4
#    'red',      # 6
#    'violet',      # 5
#    'maroon'       # 7
#]
cmap_colors = [
    'lightgreen',      # 0
    'violet',      # 1
    'red',      # 2
    'teal',      # 3
    'maroon', # 4
]
cmap = mcolors.ListedColormap(cmap_colors)
print(cmap.N)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

legend_labels = {
    0: 'true==0 & pred==true',
    1: 'true!=0 & pred==0',
    2: 'true==0 & pred!=0',
    3: 'true!=0 & pred==true',
    4: 'true!=0 & pred!=true',
}


test_genes = ['arcZ']*12 + ['gcvB']*6 + ['micA']*3 + ['ryhB']*7

' heatmap 1 '
#for s, matrix in [('KB_deduction',diff_deduction), ('neural_pred',diff_pseudo)]:#, ('pred',diff_pred)]:
#    plt.figure(figsize=(12, 8))
#    heatmap = sns.heatmap(matrix, 
#                         cmap=cmap, 
#                         norm=norm,
#                         annot=False, 
#                         #linecolor='white',
#                         #linewidths=.2,
#                         fmt="d",
#                         cbar=False)
#    heatmap.set_yticklabels([v for i,v in enumerate(test_genes)], rotation=0)
#    
#    # Create custom legend
#    #legend_labels = {
#    #    -1: 'Not a Regulator',
#    #    0: 'Not regulated',
#    #    1: 'Consistent with KB',
#    #    2: 'Dual Regulation (0 in data)',
#    #    3: 'Dual Regulation (1 in Data)',
#    #    4: 'Dual Regulation (-1 in Data)',
#    #    5: 'Inconsist (Missing in KB)',
#    #    6: 'Inconsist (Missing in Data)',
#    #    7: 'Inconsist (Reversed)'
#    #}
#    patches = [Patch(color=cmap_colors[i], label=legend_labels[i]) for i in range(0, 5)]
#    plt.legend(handles=patches, 
#               bbox_to_anchor=(1.05, 1),
#               loc='upper left', 
#               title='Value Meanings')
#    
#    
#    #sns.heatmap(jiff_matrix, annot=False, cmap='viridis')
#    plt.title('Inconsistency with Regulation')
#    plt.xlabel('Genes')
#    plt.ylabel('Overexpression')
#    plt.tight_layout()
#    
#    
#    plt.savefig(f'data_anal/heatmap_{s}.png', dpi=600)
#    plt.show()

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

plt.figure(figsize=(20, 16))

# 创建上三角元素
patches_upper = []
values_upper = []
for i in range(diff_deduction.shape[0]):
    for j in range(diff_deduction.shape[1]):
        triangle = Polygon([[j, i], [j+1, i], [j+1, i+1]], closed=True)
        patches_upper.append(triangle)
        values_upper.append(diff_deduction[i,j])

# 创建下三角元素
patches_lower = []
values_lower = []
for i in range(diff_deduction.shape[0]):
    for j in range(diff_deduction.shape[1]):
        triangle = Polygon([[j, i], [j, i+1], [j+1, i+1]], closed=True)
        patches_lower.append(triangle)
        values_lower.append(diff_pseudo[i,j])

# 添加上三角
pc_upper = PatchCollection(patches_upper, cmap=cmap, norm=norm, alpha=0.8)
pc_upper.set_array(np.array(values_upper))
plt.gca().add_collection(pc_upper)

# 添加下三角
pc_lower = PatchCollection(patches_lower, cmap=cmap, norm=norm, alpha=0.8)
pc_lower.set_array(np.array(values_lower))
plt.gca().add_collection(pc_lower)

plt.xlim(0, Y_deduction.shape[1])
plt.ylim(0, Y_deduction.shape[0])
plt.gca().invert_yaxis()
#plt.colorbar(pc_upper, label='Upper Triangle Values')
#plt.colorbar(pc_lower, label='Lower Triangle Values')
plt.title("Triangle Patch Heatmap")

patches = [Patch(color=cmap_colors[i], label=legend_labels[i]) for i in range(0, 5)]
plt.legend(handles=patches, 
           bbox_to_anchor=(1.05, 1),
           loc='upper left', 
           title='Value Meanings')


#plt.savefig(f'data_anal/heatmap_sra_deduce_vs_pseudo.png', dpi=600)
plt.show()
