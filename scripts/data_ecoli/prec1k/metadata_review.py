import pandas as pd
meta_df = pd.read_csv('dataset/prec1k/metadata.csv')

domain = {}

for col in meta_df.columns:
    dom_col = set()
    for v in meta_df[col]:
        dom_col.add(v)

    domain.update({col:dom_col})

for k,v_set in domain.items():
    if len(v_set) > 200:
        continue

    print(f'{k}: {len(v_set)}\n----------')
    for v in v_set:
        print(f'{v}\t')
    print('\n')
