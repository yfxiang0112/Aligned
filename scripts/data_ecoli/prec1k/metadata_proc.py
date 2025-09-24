import pandas as pd
import numpy as np
from sqlalchemy import all_
import json

metadata_orig = pd.read_csv('dataset/raw/metadata_precise1k_orig.csv', index_col=0)

metadata_sel = metadata_orig[['deleted_genes', 'mutated_genes', 'overexpressed_genes',
                              'Strain',
                              'Base Media',
                              'Temperature (C)', 'pH',
                              'Carbon Source (g/L)', 'Nitrogen Source (g/L)',
                              'Electron Acceptor', 'Supplement',
                              'Antibiotic for selection',
                              'Growth Rate (1/hr)']]

conditions = metadata_orig.apply(lambda x: (x['Strain'],
                                             x['Base Media'],
                                             x['Temperature (C)'], x['pH'],
                                             x['Carbon Source (g/L)'], x['Nitrogen Source (g/L)'],
                                             x['Electron Acceptor'], x['Supplement']), axis=1)
metadata_sel.insert(3,'conditions',conditions)


perturbations = metadata_sel[~metadata_sel['deleted_genes'].isna() | ~metadata_sel['mutated_genes'].isna() | ~metadata_sel['overexpressed_genes'].isna()]
perturbations = perturbations[(perturbations['deleted_genes'].apply(lambda x: len(x.split('&')) if type(x)==str else 0)<10)] # NOTE tmp
controls = metadata_sel[metadata_sel['deleted_genes'].isna() & metadata_sel['mutated_genes'].isna() & metadata_sel['overexpressed_genes'].isna()]


' condition vectors in pertur df, & missing / hitting keys in control df '
condition_pert = set(perturbations['conditions'])
condition_missing = condition_pert - set(controls['conditions'])
condition_control = condition_pert.intersection(set(controls['conditions']))

control_groups = controls[controls['conditions'].isin(condition_control)].groupby('conditions').groups
for cond in condition_missing:
    key = (cond[0], 'M9', 37, 7.0, 'glucose(2)', 'NH4Cl(1)', 'O2', np.nan)
    if key not in control_groups:
        key = ('MG1655', 'M9', 37, 7.0, 'glucose(2)', 'NH4Cl(1)', 'O2', np.nan)
    control_groups[cond] = control_groups[key]

#for k,v in control_groups.items():
#    print(f'{k}:\t{v}')

ctr_groups = perturbations['conditions'].apply(lambda x: list(control_groups[x]))
perturbations.insert(0, 'control_idx', ctr_groups)

gene_idx = pd.read_csv('dataset/gene_idx.csv', index_col=0)
with open('dataset/raw/genome_aliases.json', 'r') as f:
    gene_aliases = json.load(f)

missing_genes = set()
all_genes = set()
pert_locus = []
genome = {k:v for k,v in zip(gene_idx['symbol'],gene_idx['locus'])}
for idx,row in perturbations.iterrows():
    pert_locus.append({})
    gene_list = []

    def handle_pert(g:str, pert:int):
            gene_list.append(g)
            if g in genome:
                locus = genome[g]
                pert_locus[-1][locus] = pert
                all_genes.add(locus)
            elif g in gene_aliases:
                for locus in gene_aliases[g]['locus']:
                    pert_locus[-1][locus] = pert
                    all_genes.add(locus)
            else:
                missing_genes.add(g)

    if type(row['deleted_genes']) == str:
        for g in row['deleted_genes'].split('&'):
            handle_pert(g, -1)
    if type(row['mutated_genes']) == str:
        for g in row['mutated_genes'].split('&'):
            handle_pert(g, -1)
    if type(row['overexpressed_genes']) == str:
        for g in row['overexpressed_genes'].split('&'):
            handle_pert(g, 1)

perturbations.insert(0, 'perturbation', pert_locus)
print(perturbations)
print(f'control with no repl: {len([k for k,v in control_groups.items() if len(v)<=1])}\n')
perturbations.to_csv('dataset/precise1k/metadata.csv')


print(missing_genes, len(missing_genes))
print(len(all_genes))

pre1k_regulators = pd.DataFrame({'locus':list(all_genes), 'symbol':gene_idx.set_index('locus').loc[list(all_genes),'symbol']})
pre1k_regulators.to_csv('data_anal/list_expe_genes/pre1k_regulators.csv', index=False)
