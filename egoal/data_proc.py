import numpy as np
import pandas as pd
import random
import json

random.seed(42)

gene_mapping = pd.read_csv('dataset/raw/genome_annotations.tsv', sep='\t')
labeled_input = pd.read_csv('dataset/metadata.csv')
gene_names = list(gene_mapping['Symbol'])

df_lfc = pd.read_csv('dataset/logfc.csv', index_col=0)
df_pv = pd.read_csv('dataset/pvalue.csv', index_col=0)
df_metadata = pd.read_csv('dataset/metadata.csv', index_col=0)

df_crossval = pd.read_csv('data_anal/logs/crossval.tsv', sep='\t')
df_crossval_bef = df_crossval[df_crossval['type']=='f1_0'].reset_index(drop=True).drop(['type'],axis=1)
df_crossval_aft = df_crossval[df_crossval['type']=='f1_2'].reset_index(drop=True).drop(['type'],axis=1)

threshold = df_pv.mean(axis=1)

df_label = pd.DataFrame(0, index=df_lfc.index, columns=df_lfc.columns)
df_label[(df_lfc > .1) & (df_pv < .05)] = 1
df_label[(df_lfc < -.1) & (df_pv < .05)] = -1
#df_label[(df_lfc > 0) & (df_pv.apply(lambda r: r < threshold[r.name]/5, axis=1))] = 1
#df_label[(df_lfc < 0) & (df_pv.apply(lambda r: r < threshold[r.name], axis=1))] = -1

print(pd.concat([((df_label==1).sum(axis=1)/df_label.shape[1]), ((df_label==-1).sum(axis=1)/df_label.shape[1])],axis=1))
#print(f'-1: {(df_label==-1).sum().sum()/(df_label.shape[0]*df_label.shape[1])}\n0: {(df_label==0).sum().sum()/(df_label.shape[0]*df_label.shape[1])}\n1:{(df_label==1).sum().sum()/(df_label.shape[0]*df_label.shape[1])}')
#exit()

' align input index with label index '
df_label = df_label.loc[df_metadata.index,:]

Y_p_num = (df_label==1).sum(axis=0) / len(df_label)
Y_n_num = (df_label==-1).sum(axis=0) / len(df_label)
Y_z_num = (df_label==0).sum(axis=0) / len(df_label)

Y_prop = np.stack([Y_n_num.to_numpy(), Y_z_num.to_numpy(), Y_p_num.to_numpy()])
#print(Y_prop)
Y_max = np.argmax(Y_prop, axis=0)-1
#print(Y_max)
Y_max = {i:int(v) for i,v in enumerate(Y_max)}
with open('dataset/Y_max.json', 'w') as f:
    json.dump(Y_max, f, indent=4)
#print(sum(Y_z_num < .4))


#wt_mix_row = ((Y_p_num - Y_n_num > .1) & (Y_z_num<.6)).astype(int)\
#                        - ((Y_n_num - Y_p_num > .05) & (Y_z_num<.6)).astype(int)

wt_mix_row = ((((df_crossval_aft<df_crossval_bef).sum(axis=0)==0) & ((df_crossval_aft>df_crossval_bef).sum(axis=0)>0)) | ((df_crossval_aft>df_crossval_bef).sum(axis=0) >= 4)).astype(int)

#wt_mix_row = (Y_p_num - Y_n_num > .2).astype(int) - (Y_n_num - Y_p_num > .05).astype(int)
#wt_mix_row = (Y_z_num<.4).astype(int)
wt_zero_row = pd.Series([0]*df_label.shape[1], index=df_label.columns)
print(wt_zero_row)
#print(wt_zero_row)

idx_lst = [i for i,v in enumerate(wt_mix_row) if v!=0]
#print(len(idx_lst), idx_lst)

df_label = df_label.reset_index(drop=True)
orig_idx = df_label.index.to_list()
new_rows = [wt_mix_row]*0 + [wt_zero_row]*0
insert_idx = sorted(random.sample(range(len(df_label) + 1), len(new_rows)))

# Insert new rows
df_label_orig = df_label.copy()
for i, pos in enumerate(insert_idx):
    #df_label.insert(pos, value=new_rows[i])
    df_label = pd.concat([
        df_label.iloc[:pos], 
        pd.DataFrame([new_rows[i]]), 
        df_label.iloc[pos:]
    ]).reset_index(drop=True)


# Create index mapping
idx_map = {}
new_idx_cnt = 0

for orig_pos in range(len(orig_idx) + len(new_rows)):
    if insert_idx and new_idx_cnt == insert_idx[0]:
        # This is an inserted row - skip mapping
        #insert_idx.pop(0)
        new_idx_cnt += 1
    else:
        # This is an original row - create mapping
        actual_original = new_idx_cnt - len([p for p in insert_idx if p < new_idx_cnt])
        idx_map[actual_original] = orig_pos
        new_idx_cnt += 1

assert not df_label.isna().any().any()
print(df_label)
print(insert_idx)
assert all(df_label.loc[idx_map.values()].reset_index(drop=True) == df_label_orig)
with open('dataset/idx_mapping.json', 'w') as f:
    json.dump(idx_map, f, indent=4)



''' input gene name to matrix '''
# Create an empty one-hot matrix
data_labeled = np.zeros((len(df_label), len(gene_names)), dtype=np.int8)

# keep the list of labeled gene index
labeled_genes = set()

# Populate the matrix
for i, value in enumerate(labeled_input['overexpression']):
    if value in gene_names:
        j = gene_names.index(value)  # Get index of the value in the categories list
        labeled_genes.add(j)
        data_labeled[idx_map[i], j] = 1  # Set the corresponding position to 1


np.save('dataset/X_label.npy', data_labeled)
np.save('dataset/Y_label.npy', df_label.to_numpy())

label_set =  pd.read_csv('dataset/label_set.csv')
df_train = df_label.iloc[:,label_set['matrix_idx']]
np.save('dataset/Y_train.npy', df_train.to_numpy())
