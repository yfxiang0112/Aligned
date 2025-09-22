import json
import pandas as pd

orig = json.load(open('scripts/net_eval/results/kegg_orig_norman.json', 'r'))
df_orig = pd.Series({k:v['AUPRC_pos'] for k,v in orig.items()})

test = {}
for i in range(3):
    d = json.load(open(f'scripts/net_eval/results/kegg_norman_abl0_{i+1}.json', 'r'))
    test[str(i)] = {k:v['AUPRC_pos'] for k,v in d.items()}
df_test = pd.DataFrame(test)

df = pd.concat({'orig':df_orig, 'refined_mean':df_test.mean(axis=1)}, axis=1)

df_improved = df[df['orig'] < df['refined_mean']]
df_improved.to_csv('scripts/net_eval/dixit_pathways.csv')

print(f'P-R increased pathways: {len(df_improved)}\n',df_improved)

print('P-R dropped pathways:', len(df[df['orig'] > df['refined_mean']]))
print('P-R no change pathways:', len(df[df['orig'] == df['refined_mean']]))

#improved = set(k for k in test[0].keys() if test[0][k]['AUPRC_true']>orig[k]['AUPRC_true'])
#for d in test[1:]:
#    improved = improved.intersection(set(k for k in d.keys() if d[k]['AUPRC_true']>orig[k]['AUPRC_true']))

#print(improved)
