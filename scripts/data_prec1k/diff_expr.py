import os
import subprocess
from sys import meta_path
import pandas as pd
from pandas._libs.algos import diff_2d
from tqdm import tqdm
import json
import numpy as np
from scipy.stats import ttest_ind

count_file = "dataset/precise1k/counts.csv"
metadata_file = "dataset/precise1k/metadata.csv"
logfc_file = "dataset/precise1k/logfc.csv"
pvalue_file = "dataset/precise1k/pvalue.csv"

counts_df = pd.read_csv(count_file, index_col=0)
metadata_df = pd.read_csv(metadata_file, index_col=0)


''' logfold matrix '''

' load all control (WT) rows '
#ref_idx_list = list(metadata_df[metadata_df['overexpression'].apply(lambda x: 'WT' in x)].index)

' for all groups: '
fc_df = []
p_value_df = {'index':counts_df.columns}
fitness_dict = {}

for idx,row in tqdm(metadata_df.iterrows(),desc='computing lfc & pv', total=len(metadata_df)):


    ' build reference row (mean of control rows in group)'
    ref_idx = eval(row['control_idx'])
    ref_df = counts_df.loc[ref_idx, :].astype(int)
    ref_row = ref_df.mean(axis=0).replace(0, 1e-6)
    if len(ref_df) <= 2:
        ref_df_ = (ref_df - 5).replace([-5,-4,-3,-2,-1],0)
        ref_df_.index = ref_df.index+'_'
        ref_df = pd.concat([ref_df+5, ref_df_], axis=0).replace(-1,0)

    ' compute fold change '
    #for acc_lst in exr_lst:
    #for g,acc_lst in exr_groups.items():
    exr_df = counts_df.loc[[idx], :].astype(int)
    #if len(exr_df) <= 1:
    exr_df_ = (exr_df - 5).replace([-5,-4,-3,-2,-1],0)
    exr_df_.index = exr_df.index+'_'
    exr_df = pd.concat([exr_df+5, exr_df_], axis=0).replace(-1,0)
    exr_row = exr_df.mean(axis=0).replace(0, 1e-6)

    fitness_dict[idx] = exr_df.mean().mean() / ref_df.mean().mean()

    #row_ex.replace(0,1,inplace=True)
    fc_df.append( pd.DataFrame({idx:(exr_row / ref_row).replace(0,1e-6)}) )
    p_value_df[idx] = []

    for gene in ref_df.columns:

        ''' Perform t-test assuming independent samples '''
        t_stat, p_value = ttest_ind(ref_df.loc[:,gene].values, exr_df.loc[:,gene].values, equal_var=False)
        p_value_df[idx].append(p_value)
        #if np.isnan(p_value):
        #    print(ref_df.loc[:,gene].values, exr_df.loc[:,gene].values, p_value)
        #    exit()
        #print(np.array(p_value_df[acc_lst[0]]))
    #if np.isnan(np.array(p_value_df[acc_lst[0]])).any():
    #    print(ref_df)
    #    print(exr_df)
    #    exit()
        
with open('dataset/precise1k/fitness.json', 'w') as f:
    json.dump(fitness_dict, f, indent=4)

' get logfc & p-value matrix '
logfc_df = np.log2(pd.concat(fc_df, axis=1))
logfc_df = logfc_df.transpose().sort_index()#.loc[metadata_df.index]
logfc_df.to_csv(logfc_file)

p_value_df = pd.DataFrame(p_value_df).set_index('index')
p_value_df = p_value_df.fillna(1.).transpose().sort_index()#.loc[metadata_df.index]
p_value_df.to_csv(pvalue_file)

print(logfc_df)
print(p_value_df)
