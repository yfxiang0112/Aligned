import pandas as pd
import numpy as np
import json

#df = pd.read_csv('dataset/raw/gene_synonym.tsv', sep='\t')
#alias_df = pd.DataFrame()
#alias_df['Locus'] = df['Aliases'].map(lambda x: x[:5] if type(x)==str else np.NaN)
#alias_df['Symbol'] = df['Symbol']
#alias_df['Aliases'] = df['Aliases'].map(lambda x: set(x.split(', ')[2:])if type(x)==str else np.NaN)
#alias_df['Aliases'] = alias_df['Aliases'].map(lambda x: x if type(x)==set and len(x)>0 else np.NaN)
#alias_df.dropna(axis=0, inplace=True)

alias_df = pd.read_csv('dataset/raw/genome_aliases.csv')
alias_dict = {}
for idx,row in alias_df.iterrows():
    for alias in eval(row['Aliases']):
        if alias not in alias_dict:
           alias_dict[alias] = {'locus':[row['Locus']], 'symbol':[row['Symbol']]}
        else:
           alias_dict[alias]['locus'].append(row['Locus'])
           alias_dict[alias]['symbol'].append(row['Symbol'])
alias_df = alias_df.set_index('Locus').sort_index()
alias_df.to_csv('dataset/raw/genome_aliases.csv', index=True)
with open('dataset/raw/genome_aliases.json','w') as f:
    json.dump(alias_dict, f, indent=4)
