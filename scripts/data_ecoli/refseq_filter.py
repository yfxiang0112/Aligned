'''Opens the fasta file, reads the sequences, and filters them based on the presence of any string from the list in the identifier.'''
import pandas as pd
import re
import json

fasta_file = 'dataset/raw/ecoli_refseq/ecoli_cds.fna'
output_file = 'dataset/raw/ecoli_refseq/ecoli_cds_241genes.fna'
mapping_file = 'dataset/raw/locus_mapping.json'
mapping = {}

gene_df = pd.read_csv('gene_list.csv')
gene_list = list(gene_df['gene1'])

with open(fasta_file, 'r') as infile, open(output_file, 'w') as outfile:
    keep_sequence = False
    for line in infile:
        # Check if the line is a header
        if line.startswith(">"):
            # If no string in the header matches, skip this sequence
            if any(s in line for s in gene_list):
                keep_sequence = True
                outfile.write(line)

                ''' keep a mapping from sam genename to locus '''
                gene_id = re.findall(r'NC_000913.3_cds_\w\w_\d*.\d_\d*', line)[0]
                locus = re.findall(r'b\d\d\d\d', line)[0]
                mapping[gene_id] = locus
            else:
                keep_sequence = False
        elif keep_sequence:
            # If it's a sequence and the identifier is valid, write it to the output file
            outfile.write(line)

with open(mapping_file, 'w') as f:
    json.dump(mapping, f, indent=4)
