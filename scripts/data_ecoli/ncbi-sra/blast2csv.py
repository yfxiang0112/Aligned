import os
import subprocess
import pandas as pd
from pandas._libs.algos import diff_2d
from tqdm import tqdm
import json
import numpy as np
from scipy.stats import ttest_ind

sam_dir = "dataset/raw/blast_res"
metadata_df = pd.read_csv("dataset/metadata.csv", index_col=0)
mapping_file = "dataset/raw/locus_mapping.json" # map NCBI gene accessions to locus tags
with open(mapping_file, 'r') as f:
    locus_mapping = json.load(f)

count_file = "dataset/counts.csv"
logfc_file = "dataset/logfc.csv"
pvalue_file = "dataset/pvalue.csv"

' dict for group-wise logfc computation '
df_dict = {}
df_list = []

''' order of genome annotation
    (align with KB matrix) '''
gene_output_241 = pd.read_csv('dataset/genes_241.csv')
index_sorted = list(gene_output_241['locus'])



' for all sam files: '
for accession in tqdm(metadata_df.index):
    filename = f'{accession}.sam'
    file_path = os.path.join(sam_dir, filename)
    if os.path.isfile(file_path):

        ' .sam file 2 counts '
        result = subprocess.run(f"samtools view -SF 4 {file_path} |perl -alne '{{$h{{$F[2]}}++}}END{{print \"$_\t$h{{$_}}\" foreach sort keys %h }}'", capture_output=True, text=True, shell=True)
        
        ' result tsv 2 df '
        if result.returncode == 0:
            data = result.stdout.strip().splitlines()
            df = pd.DataFrame([line.split('\t') for line in data])
            
            ' rename cols & fill NaNs & align index with KB '
            df.columns = ['accession', accession]
            df['accession'] = df['accession'].map(lambda x: locus_mapping[x])
            df.set_index('accession', inplace=True)
            df = pd.concat([df, pd.DataFrame(0, index=[i for i in index_sorted\
                                            if i not in df.index], columns=df.columns)]).reindex(index_sorted)

            ' build group-wise dict '
            if accession[:-3] not in df_dict:
                df_dict[accession[:-3]] = [df]
            else:
                df_dict[accession[:-3]].append(df)
            df_list.append(df)



assert df_dict

''' count matrix '''
final_df = pd.concat(df_list, axis=1)
' Concatenate all dfs along columns (axis=1) & save '
final_df = final_df.transpose().sort_index()
final_df.to_csv(count_file)




''' logfold matrix '''

' load all control (WT) rows '
ref_idx_list = list(metadata_df[metadata_df['overexpression'].apply(lambda x: 'WT' in x)].index)

' for all groups: '
fc_df = []
p_value_df = {}
for key,lst in df_dict.items():
    if 'accessions' not in p_value_df:
        p_value_df['accessions'] = lst[0].index

    ' build reference row (mean of control rows in group)'
    ref_df = [col.astype(int) for col in lst if col.columns[0] in ref_idx_list]
    if len(ref_df) == 0:
        continue
    ref_df = pd.concat(ref_df, axis=1)
    ref_row = ref_df.mean(axis=1).replace(0, 1e-6)

    ' compute fold change '
    group = pd.DataFrame({row.columns[0]:(row.transpose().astype(int).apply(lambda x:int(x)) /
                             ref_row).replace(0,1e-6) for row in lst})
    fc_df.append(group)
    

    for row in lst:
        col = row.columns[0]
        if col in ref_idx_list:
            p_value_df[col] = [1.]*len(ref_df.index)
            continue
        row_ex = pd.concat([row.astype(int)+1, row.astype(int)-1], axis=1).replace(-1,0)
        p_value_df[col] = []
        for gene in ref_df.index:
            #control_values = 
            #experiment_value = row.loc[gene].values
            ''' Perform t-test assuming independent samples '''
            t_stat, p_value = ttest_ind(ref_df.loc[gene].values, row_ex.loc[gene].values, equal_var=False)
            p_value_df[col].append(p_value)
        

' get logfc & p-value matrix '
logfc_df = np.log2(pd.concat(fc_df, axis=1))
logfc_df = logfc_df.transpose().sort_index()
logfc_df.to_csv(logfc_file)

p_value_df = pd.DataFrame(p_value_df).set_index('accessions')
p_value_df = p_value_df.transpose().sort_index()
p_value_df.to_csv(pvalue_file)

print(logfc_df)
print(p_value_df)

