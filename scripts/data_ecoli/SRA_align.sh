#!/bin/bash

# Input file containing list of strings
accessions="dataset/raw/accessions_3"
n_threads=8

# Define parallel execution
cat "$accessions" | xargs -I {} -P $n_threads bash -c $'
  ~/yfxiang/bioinfo/ncbi-magicblast-1.7.2/bin/magicblast -sra {} -db dataset/raw/ecoli_refseq/ecoli_cds -no_unaligned -num_threads 4 > dataset/raw/blast_res/{}.sam
	samtools view -SF 4 dataset/raw/blast_res/{}.sam | perl -alne \'{$h{$F[2]}++}END{print "$_\t$h{$_}" foreach sort keys %h }\' > dataset/raw/blast_res/{}_count.txt
	/bin/rm dataset/raw/blast_res/{}.sam
  echo "completed {}"
  wait
'
