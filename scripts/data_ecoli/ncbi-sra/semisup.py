import pandas as pd
import numpy as np

growth_delay_path = 'dataset/raw/overexpression_growth_delay.xlsx'
gene_annotation_path = 'dataset/raw/genome_annotations.tsv'
output_path = 'dataset/X_semisup.csv'
alias_path = 'dataset/raw/genome_aliases.csv'

# Read the Excel file
excel_df = pd.read_excel(growth_delay_path)
mapping_df = pd.read_csv(gene_annotation_path, sep='\t')
alias_df = pd.read_csv(alias_path)
print(excel_df)


# Create a mapping dictionary from the mapping DataFrame
locus_dict = dict(zip(mapping_df['Symbol'], mapping_df['Locus tag']))
index_dict = dict(zip(mapping_df['Symbol'], mapping_df.index))

alias_dict = {}
for _,row in alias_df.iterrows():
    for a in eval(row['Aliases']):
        alias_dict[a] = row['Symbol']

excel_df['Gene name'] = excel_df['Gene name'].map(lambda x: alias_dict[x] if x in alias_dict else x)
for g in excel_df['Gene name']:
    if g not in locus_dict and g not in alias_dict:
        print(g)
exit()


# Convert strings in the first column using the mapping
semisup_df = pd.DataFrame()
semisup_df['index'] = excel_df['Gene name'].map(index_dict)
semisup_df['locus'] = excel_df['Gene name'].map(locus_dict)


# Divide values in the second column by 2
semisup_df['growth'] = 1 / excel_df['Delay factor']

#TODO
semisup_df.dropna(axis=0, inplace=True)

semisup_df['index'] = semisup_df['index'].astype(int)
semisup_df.set_index('index', inplace=True)
semisup_df.sort_index(inplace=True)


# Save the result as CSV
semisup_df.to_csv(output_path, index=True)

print(semisup_df)




''' generate unlabel dataset '''
n_unlabel = len(semisup_df)
data_unlabel = np.zeros((n_unlabel, len(mapping_df)), dtype=np.int8)
for i, gene_idx in enumerate(semisup_df.index):
    data_unlabel[i,gene_idx] = 1
np.save('dataset/X_unlabel.npy', data_unlabel)
print('unlabel', data_unlabel.shape)
