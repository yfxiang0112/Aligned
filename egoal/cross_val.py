import pandas as pd
import numpy as np
import json
from datetime import datetime

from abl import abduce

X = np.load('dataset/X_label.npy')
Y = np.load('dataset/Y_train.npy')
#X_unlabel = np.load('dataset/X_unlabel.npy')
#z_groundtruth = pd.read_csv('dataset/X_semisup.csv')['growth'].to_numpy()
log_file = f'log/crossval-{datetime.now()}.txt'.replace(' ','-')

metadata = pd.read_csv('dataset/metadata_gene.csv', index_col=0)
regulators = ["arcZ","cyaR","gcvB","micA","ryhB","rydC"]
with open('dataset/idx_mapping.json', 'r') as f:
    idx_mapping = json.load(f)



for idx, (gene, row) in enumerate(metadata.iterrows()):
    if gene not in regulators:
        continue

    idx_lst = eval(row['dataset_items'])
    z = -1 if row['fitness']<1 else 1
    gene_idx = int(row['matrix_idx'])

    ' construct unlabel X & z array '
    X_unlabel = np.zeros(shape=(5, X.shape[1]))
    for i in range(5):
        X_unlabel[i, gene_idx-2+i] = 1
    z_groundtruth = [0,0,z,0,0]

    ' mapping test set index '
    test_idx = [idx_mapping[str(i)] for i in idx_lst]

    with open(log_file, 'a') as f:
        f.write(f'{"="*40}\n{"="*40}\n{idx+1}/{len(metadata)}: {gene} overexpression\n{"-"*40}')
    print(f'Cross validation: test on {gene}')

    kb_weights = [(1,0), (1,0), (.5,.5), (0,1), (0,1)]
    abduce(X=X, Y=Y, X_unlabel=X_unlabel, z_groundtruth=z_groundtruth,
           pos_trn_pth='rules/pos_regu',
           neg_trn_pth='rules/neg_regu',
           pos_gem_pth='rules/pos_gem',
           neg_gem_pth='rules/neg_gem',
           gem_annotation_pth='rules/gem_annot',
           T=5,
           max_modify=20,
           budget=5000,
           pretrain_epc=50,
           pretrain_lr=0.001,
           subset_threshold=1.,
           retrain_epc=50,
           retrain_lr=0.01,
           seed=42, test_idx=test_idx, kb_weight=kb_weights, log_file=log_file)

    with open(log_file, 'a') as f:
        f.write('\n\n\n')
