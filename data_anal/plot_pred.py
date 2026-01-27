import pandas as pd

annotation_df = pd.read_csv('dataset/raw/genome_annotations.tsv', sep='\t')
gene_mapping_df = pd.read_csv('dataset/genes_241.csv', index_col=0)
gene_lst_df = annotation_df.iloc[gene_mapping_df['matrix_idx'],:]
gene_lst_df.index = gene_mapping_df.index

idx_list = [42,43]

#metadata = pd.read_csv('dataset/metadata.csv')
#print(metadata.groupby('overexpression').indices)

pred_df = pd.read_csv('data_anal/predictions.csv', index_col=0)
pred_df = pd.concat([gene_lst_df['Locus tag'],gene_lst_df['Symbol'],pred_df],axis=1)

pvalues = []
logfcs = []
df_pvalue = pd.read_csv('dataset/pvalue.csv', index_col=0)
for idx in idx_list:
    pvalues.append(df_pvalue.iloc[idx,:])
df_logfc = pd.read_csv('dataset/logfc.csv', index_col=0)
for idx in idx_list:
    logfcs.append(df_logfc.iloc[idx,:])

for idx,pval,lfc in zip(range(len(pvalues)), pvalues, logfcs):
    pval.index = pred_df.index
    lfc.index = pred_df.index
    pred_df[f'pvalue{idx}'] = pval
    pred_df[f'logfc{idx}'] = lfc

df_incorrect = pred_df[pred_df['bef_f1'] > pred_df['aft_f1']]
df_incons = df_incorrect[(df_incorrect['pvalue0'] >= .5) | (df_incorrect['pvalue1'] >= .5) | (df_incorrect['aft_pred0']*df_incorrect['logfc0'] <= 0) | (df_incorrect['aft_pred1']*df_incorrect['logfc1']<=0)]


print(f'Labels improved by ABL: {sum(pred_df["bef_f1"]<pred_df["aft_f1"])}')
print(f'Incorrect predictions: {len(df_incorrect)}')
print(f'Inconsistent labels with groundtruth: {len(df_incons)}\n')
print(df_incons)
