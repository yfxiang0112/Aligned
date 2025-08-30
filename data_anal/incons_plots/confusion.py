import numpy as np
import torch
from scipy.sparse import load_npz
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from egoal.reasoner import RegualtoryKB


data_name = 'norman'
kb_name = 'dorothea'

device = 'cuda'

Y = load_npz(f'dataset/human/{data_name}_Y.npz').toarray()
X = load_npz(f'dataset/human/{data_name}_X.npz').toarray()


if os.path.exists(f'data_anal/incons_plots/{data_name}_Y_d.npy'):
    Y_deduction = np.load(f'data_anal/incons_plots/{data_name}_Y_d.npy')
else:
    KB = RegualtoryKB(pos_trn_pth=f'rules/human/{data_name}_KB_P.npz', neg_trn_pth=f'rules/human/{data_name}_KB_N.npz', device=device)
    KB.closure_(T=5, closure_type='weighted')
    
    Y_deduction = KB.deduce(torch.tensor(X).float().to(device)).to('cpu').numpy()
    np.save(f'data_anal/incons_plots/{data_name}_Y_d.npy', Y_deduction)

classes =[-1, 0,1] 
n_classes = 3
flat_y_t, flat_y_d = Y.flatten(), Y_deduction.flatten()
confusion = confusion_matrix(flat_y_t, flat_y_d, labels=classes)
confusion = confusion / np.sum(confusion)


#if normalize:
#    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    fmt = '.2f'
#else:
#    fmt = 'd'

# Create figure
# Create custom colormap for TP, TN, FP, FN

tp_cmap = plt.cm.Greens
fp_cmap = plt.cm.Reds
fn_cmap = plt.cm.Oranges

# Create a matrix to store colors
colored_matrix = np.zeros((n_classes, n_classes, 4))

# Get normalization ranges
#max_tp = np.max(np.diag(confusion)) if np.max(np.diag(confusion)) > 0 else 1
#max_error = np.max(confusion - np.diag(np.diag(confusion))) if np.max(confusion - np.diag(np.diag(confusion))) > 0 else 1

colored_matrix[0,0] = tp_cmap(10 * confusion[0,0])
colored_matrix[1,1] = [.5,.5,.5,1.]
colored_matrix[2,2] = tp_cmap(10 * confusion[2,2])

colored_matrix[0,1] = fn_cmap(10 * confusion[0,1])
colored_matrix[2,1] = fn_cmap(10 * confusion[2,1])

colored_matrix[1,0] = fp_cmap(10 * confusion[1,0])
colored_matrix[2,0] = fp_cmap(10 * confusion[2,0])
colored_matrix[0,2] = fp_cmap(10 * confusion[0,2])
colored_matrix[1,2] = fp_cmap(10 * confusion[1,2])
#for i in range(n_classes):
#    for j in range(n_classes):
#        value = confusion[i, j]
#        
#        if i == j:  # TP
#            rgba = tp_cmap(value)
#        else:
#            rgba = fp_cmap(value)
#        
#        colored_matrix[i, j] = rgba
    
figsize=(8, 6)

fig, ax = plt.subplots(figsize=figsize)

# Plot using imshow
im = ax.imshow(colored_matrix, interpolation='nearest', aspect='auto')
#ax.figure.colorbar(im, ax=ax)


# Set labels
if classes is None:
    classes = np.unique(np.concatenate([y_true, y_pred]))

# Set ticks and labels
ax.set(xticks=np.arange(confusion.shape[1]),
       yticks=np.arange(confusion.shape[0]),
       xticklabels=classes,
       yticklabels=classes,
       title=f'Confusion Matrix between {data_name} Dataset & {kb_name} KB Deduction Results',
       ylabel='Label in Dataset',
       xlabel='Label in KB Deduction Result')

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
thresh = confusion.max() / 2.
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(j, i, format(confusion[i, j]*100, '.2f')+'%',
                ha="center", va="center",
                color="white" if confusion[i, j] > thresh else "black")

legend_elements = [
    mpatches.Patch(facecolor=tp_cmap(0.5), edgecolor='black', label='Consistent Instances'),
    mpatches.Patch(facecolor=fp_cmap(0.5), edgecolor='black', label='Incorrect Instances in Data'),
    mpatches.Patch(facecolor=fn_cmap(0.5), edgecolor='black', label='Incomplete Instances in KB')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))

plt.tight_layout()
plt.savefig(f'data_anal/incons_plots/{data_name}_confusion.png', dpi=600)
plt.show()
