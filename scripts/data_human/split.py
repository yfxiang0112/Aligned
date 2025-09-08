import pandas as pd
import numpy as np
import json

np.random.seed(42)
data_name = 'dixit'

metadata = pd.read_csv(f'dataset/human/{data_name}_metadata.csv',index_col=0)
regulatory = pd.read_csv('rules/human/regulatory_dorothea.csv', index_col=0)

# NOTE tmp ##############
#test_idx = np.random.choice([True, False], size=len(metadata), p=[.2, .8])
#test_idx = test_idx & (metadata['pert'].apply(lambda x: len(eval(x))==2))
#test = metadata[test_idx]
#train = metadata[~test_idx]

test = pd.read_csv(f'dataset/human/{data_name}_test_set.csv', index_col=0)
train = metadata.loc[~ metadata['pert'].isin(test['pert'])]
# NOTE end ##############

print('test metadata:\n',test)
test.reset_index(inplace=True, drop=True)

' data idx of test set '
test_data_idx = sum(metadata[metadata['pert'].isin(test['pert'])].apply(lambda x: list(range(x['data_start_idx'],x['data_end_idx+1'])), axis=1), [])
print('test dataset len: ',len(test_data_idx))
np.save(f'dataset/human/{data_name}_test_idx.npy', test_data_idx)


'seen genes in train set '
train_genes = set(sum(train['pert'].apply(eval),[]))
train_genes.remove('ctrl')
regulators = set(regulatory['tf']).intersection(train_genes)



' 1 pert, seen in train data '
seen_1_pert = test['pert'].apply(lambda x: 
                            (eval(x)[0] == 'ctrl' and eval(x)[1] in train_genes) or\
                            (eval(x)[1] == 'ctrl' and eval(x)[0] in train_genes))

' 1 pert, unseen in train data '
unseen_1_pert = test['pert'].apply(lambda x: 
                            (eval(x)[0] == 'ctrl' and eval(x)[1] not in train_genes) or\
                            (eval(x)[1] == 'ctrl' and eval(x)[0] not in train_genes))

' 2 pert, 1 seen (or 2 seen) in train data '
seen_2_pert = test['pert'].apply(lambda x: 
                            ('ctrl' not in eval(x)) and \
                            (eval(x)[0] in train_genes or eval(x)[0] in train_genes))
#                            (eval(x)[0] not in train_genes and eval(x)[1] in train_genes) or\
#                            (eval(x)[0] not in train_genes and eval(x)[0] in train_genes))

' 2 pert, all unseen in train data '
unseen_2_pert = test['pert'].apply(lambda x: 
                            ('ctrl' not in eval(x)) and \
                            (eval(x)[0] not in train_genes and eval(x)[0] not in train_genes))


################################################

print('  seen 1 pert: ',len(test.loc[seen_1_pert]))
print('unseen 1 pert: ', len(test.loc[unseen_1_pert]))
print('  seen 2 pert: ',len(test.loc[seen_2_pert]))
print('unseen 2 pert: ', len(test.loc[unseen_2_pert]))

' build test_set type tag in csv '
test['test_type'] = ['none']*len(test)
test.loc[seen_1_pert, 'test_type'] = 'seen_1_pert'
test.loc[unseen_1_pert, 'test_type'] = 'unseen_1_pert'
test.loc[seen_2_pert, 'test_type'] = 'seen_2_pert'
test.loc[unseen_2_pert, 'test_type'] = 'unseen_2_pert'

' reset test set idx in csv '
#print(test[['data_start_idx','data_end_idx+1']])
for i in range(len(test)):
    test.loc[i, 'data_end_idx+1'] -= \
        test.loc[i, 'data_start_idx'] if i == 0\
        else (test.loc[i, 'data_start_idx'] - test.loc[i-1, 'data_end_idx+1'])

    test.loc[i, 'data_start_idx'] = 0 if i==0 else test.loc[i-1, 'data_end_idx+1']
#print(test[['data_start_idx','data_end_idx+1']])
#print(len(test_data_idx))

test.to_csv(f'dataset/human/{data_name}_test_set.csv') # NOTE

split = {}
split['train'] = list(train['pert'].apply(lambda x: f'{eval(x)[0]}+{eval(x)[1]}' if len(eval(x))>1 else eval(x)[0]))
split['test'] = list(test['pert'].apply(lambda x: f'{eval(x)[0]}+{eval(x)[1]}' if len(eval(x))>1 else eval(x)[0]))
split['val'] = split['test']
json.dump(split, open(f'dataset/human/{data_name}_split.json', 'w'), indent=4)
