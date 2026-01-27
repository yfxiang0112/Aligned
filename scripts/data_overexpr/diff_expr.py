import os
import subprocess
from sys import meta_path
import pandas as pd
from pandas._libs.algos import diff_2d
from tqdm import tqdm
import json
import numpy as np
from scipy.stats import ttest_ind

count_dir = "dataset/raw/blast_res"
metadata_df = pd.read_csv("dataset/raw/metadata_wt.csv", index_col=0)
gene_mapping_df = pd.read_csv('dataset/gene_idx.csv', index_col=0)
index_sorted = list(gene_mapping_df['locus'])

locus_mapping = {}
for idx,row in gene_mapping_df.iterrows():
    for acc in eval(row['ncbi']):
        locus_mapping[acc] = row['locus']

count_file = "dataset/ncbi-sra/counts.csv"
logfc_file = "dataset/ncbi-sra/logfc.csv"
pvalue_file = "dataset/ncbi-sra/pvalue.csv"

' dict for group-wise logfc computation '
group_dict = {}
df_list = []

''' order of genome annotation
    (align with KB matrix) '''
#gene_output_241 = pd.read_csv('dataset/genes_241.csv')
#index_sorted = list(gene_output_241['locus'])



' for all sam files: '
for accession in tqdm(metadata_df.index):
    filename = f'{accession}_count.txt'
    file_path = os.path.join(count_dir, filename)
    if os.path.isfile(file_path):

        #result = subprocess.run(f"samtools view -SF 4 {file_path} |perl -alne '{{$h{{$F[2]}}++}}END{{print \"$_\t$h{{$_}}\" foreach sort keys %h }}'", capture_output=True, text=True, shell=True)
        #
        #' result tsv 2 df '
        #if result.returncode == 0:
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
            df = pd.DataFrame([line.split('\t') for line in data])

            #TODO
            if df.empty:
                continue
            
            ' rename cols & fill NaNs & align index with KB '
            df.columns = ['accession', accession]
            df['accession'] = df['accession'].map(lambda x: locus_mapping[x])
            df = df.groupby('accession').sum()
            #df.set_index('accession', inplace=True)
            df = pd.concat([df, pd.DataFrame(0, index=[i for i in index_sorted\
                                            if i not in df.index], columns=df.columns)]).reindex(index_sorted)

            ' build group-wise dict '
            if accession[:-3] not in group_dict:
                group_dict[accession[:-3]] = [accession]
            else:
                group_dict[accession[:-3]].append(accession)
            df_list.append(df)



assert group_dict

''' count matrix '''
final_df = pd.concat(df_list, axis=1)
' Concatenate all dfs along columns (axis=1) & save '
final_df = final_df.transpose().sort_index()
final_df.to_csv(count_file)

print(final_df)

#ref = pd.read_csv('dataset/')

final_df = final_df.astype(int)
final_df = final_df.multiply(1000 / (final_df['b3067'] + final_df['b2600'] + final_df['b2699']),axis=0)



''' logfold matrix '''

' load all control (WT) rows '
#ref_idx_list = list(metadata_df[metadata_df['overexpression'].apply(lambda x: 'WT' in x)].index)

' for all groups: '
fc_df = []
p_value_df = {'accessions':final_df.columns}
fitness_dict = {}

for key,lst in tqdm(group_dict.items(),'computing lfc & pv'):

    exr_groups = {}
    ref_groups = []
    exr_lst = []
    for accession in lst:
        if metadata_df.loc[accession,'overexpression'] == 'WT':
            ref_groups.append(accession)
        elif metadata_df.loc[accession,'overexpression'] in exr_groups:
            exr_groups[metadata_df.loc[accession,'overexpression']].append(accession)
            exr_lst.append([accession])
        else:
            exr_groups[metadata_df.loc[accession,'overexpression']] = [accession]
            exr_lst.append([accession])


    ' build reference row (mean of control rows in group)'
    ref_df = final_df.loc[ref_groups,:].astype(int)
    ref_row = ref_df.mean(axis=0).replace(0, 1e-6)
    if len(ref_df) <= 2:
        ref_df_ = (ref_df - 5).replace([-5,-4,-3,-2,-1],0)
        ref_df_.index = ref_df.index+'_'
        ref_df = pd.concat([ref_df+5, ref_df_], axis=0).replace(-1,0)

    ' compute fold change '
    #for acc_lst in exr_lst:
    for g,acc_lst in exr_groups.items():
        exr_df = final_df.loc[acc_lst, :].astype(int)
        if len(exr_df) <= 1:
            exr_df_ = (exr_df - 5).replace([-5,-4,-3,-2,-1],0)
            exr_df_.index = exr_df.index+'_'
            exr_df = pd.concat([exr_df+5, exr_df_], axis=0).replace(-1,0)
        exr_row = exr_df.mean(axis=0).replace(0, 1e-6)

        fitness = exr_df.mean().mean() / ref_df.mean().mean()
        if g in fitness_dict:
            fitness_dict[g] = (fitness_dict[g]+fitness)/2
        else:
            fitness_dict[g] = fitness

        #row_ex.replace(0,1,inplace=True)
        fc_df.append( pd.DataFrame({acc_lst[0]:(exr_row / ref_row).replace(0,1e-6)}) )
        p_value_df[acc_lst[0]] = []

        for gene in ref_df.columns:

            ''' Perform t-test assuming independent samples '''
            t_stat, p_value = ttest_ind(ref_df.loc[:,gene].values, exr_df.loc[:,gene].values, equal_var=False)
            p_value_df[acc_lst[0]].append(p_value)
            #if np.isnan(p_value):
            #    print(ref_df.loc[:,gene].values, exr_df.loc[:,gene].values, p_value)
            #    exit()
        #print(np.array(p_value_df[acc_lst[0]]))
        #if np.isnan(np.array(p_value_df[acc_lst[0]])).any():
        #    print(ref_df)
        #    print(exr_df)
        #    exit()
        
with open('dataset/ncbi-sra/fitness.json', 'w') as f:
    json.dump(fitness_dict, f, indent=4)

metadata_df = pd.read_csv("dataset/ncbi-sra/metadata.csv", index_col=0)

' get logfc & p-value matrix '
logfc_df = np.log2(pd.concat(fc_df, axis=1))
logfc_df = logfc_df.transpose().sort_index()#.loc[metadata_df.index]
logfc_df.to_csv(logfc_file)

p_value_df = pd.DataFrame(p_value_df).set_index('accessions')
p_value_df = p_value_df.fillna(1.).transpose().sort_index()#.loc[metadata_df.index]
p_value_df.to_csv(pvalue_file)

print(logfc_df)
print(p_value_df)
