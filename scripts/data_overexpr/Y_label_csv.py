import pandas as pd

metadata = pd.read_csv('dataset/metadata.csv', index_col=0)
df1 = pd.read_csv('dataset/logfc.csv', index_col=0).loc[metadata.index]
df2 = pd.read_csv('dataset/pvalue.csv', index_col=0).loc[metadata.index]

# Initialize the result DataFrame with zeros
result_df = pd.DataFrame(0, index=metadata.index, columns=df1.columns)

# Apply the conditions
result_df[(df1 > .5) & (df2 < 0.05)] = 1
result_df[(df1 < .5) & (df2 < 0.05)] = -1
#result_df[df2 >= 0.05] = 0

print(result_df)
result_df.to_csv('dataset/Y_label.csv')
