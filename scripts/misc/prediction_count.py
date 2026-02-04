import pandas as pd
df = pd.read_csv('tmp.csv')

df_pred = df.loc[df['label']=='pred',:].iloc[:,1:]
df_pred.reset_index(drop=True,inplace=True)

df_test = df.loc[df['label']=='test',:].iloc[:,1:]
df_test.reset_index(drop=True,inplace=True)

FN_u = df_pred.columns[((df_pred==0)&(df_test==1)).sum(axis=0)>2].astype(int)
FP_u = df_pred.columns[((df_pred==1)&(df_test==0)).sum(axis=0)>2].astype(int)

FN_d = df_pred.columns[((df_pred==0)&(df_test==-1)).sum(axis=0)>2].astype(int)
FP_d = df_pred.columns[((df_pred==-1)&(df_test==0)).sum(axis=0)>2].astype(int)

FN = df_pred.columns[((df_pred==0)&(df_test!=0)).sum(axis=0)>2].astype(int)
FP = df_pred.columns[((df_pred!=0)&(df_test==0)).sum(axis=0)>2].astype(int)

print(len(FN_u), len(FP_u))
print(len(FN_d), len(FP_d))
print(list(FN))
print(list(FP))
print(FP[108:114])
