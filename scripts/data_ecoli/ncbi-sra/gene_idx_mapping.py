import pandas as pd
import numpy as np
import re
import cobra

from pandas._libs.lib import is_float

annotation = pd.read_csv('dataset/raw/genome_annotations.tsv', sep='\t')
metadata = pd.read_csv('dataset/metadata.csv')
semisup = pd.read_csv('dataset/X_semisup.csv')
print(annotation)
print(metadata)

ncbi_mapping = {}
with open('dataset/raw/ecoli_refseq/ecoli_cds.fna', 'r') as f:
    for line in f:
        if line.startswith(">"):
            accession = re.findall(r'NC_\d*\.\d*_cds_[\w\.]*', line)
            locus = re.findall(r'b\d\d\d\d', line)
            if len(accession)>0 and len(locus)>0:
                if locus[0] in ncbi_mapping:
                    ncbi_mapping[locus[0]].append(accession[0])
                else:
                    ncbi_mapping[locus[0]] = [accession[0]]


groups = metadata.groupby('overexpression').groups
metadata_idx = [list(groups[sym]) if sym in groups else [] for sym in annotation['Symbol']]

semisup_dict = {locus:i for i,locus in enumerate(semisup['locus'])}
semisup_idx = [semisup_dict[i] if i in semisup_dict else -1 for i in annotation['Locus tag']]
ncbi_idx = [ncbi_mapping[locus] if locus in ncbi_mapping else [] for locus in annotation['Locus tag']]

model = cobra.io.load_model('iML1515')  # Requires cobrapy and the model file
iml_dict = {v:i for i,v in enumerate(sorted([gene.id for gene in model.genes]))}
iml_list = [iml_dict[locus] if locus in iml_dict else -1 for locus in annotation['Locus tag']]

mapping = pd.DataFrame({'locus':annotation['Locus tag'], 'symbol':annotation['Symbol'], 'ncbi':ncbi_idx, 'metadata_idx':metadata_idx, 'semisup_idx':semisup_idx, 'iml1515_idx':iml_list})
print(mapping)
mapping.to_csv('dataset/gene_idx.csv', index=True)
