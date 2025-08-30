import omnipath
import pandas as pd

go = pd.read_csv('../GEARS/norman/go.csv')

interaction = omnipath.interactions.AllInteractions().get()
annot = omnipath.requests.Annotations()

symbols = {}
non_uniq = {}
empty_ann = set()
omni_genes = set(interaction['source']).union(set(interaction['target']))

for i, g in enumerate(omni_genes):
    ann = annot.get(g)
    sym_set = set(ann['genesymbol'])

    if len(sym_set) > 1:
        non_uniq[g] = sym_set
    if len(sym_set) == 0:
        empty_ann.add(g)

    symbols[g] = sym_set.pop() if len(sym_set)>0 else ''

    #if i>10:
    #    break

    if i % 100 == 0:
        print(f'processed {i} / {len(omni_genes)} items')

print(symbols)
print(len(set(symbols.values()).intersection(set(go['source']))))

interaction['source'] = interaction['source'].apply(lambda x: symbols[x])
interaction['target'] = interaction['target'].apply(lambda x: symbols[x])

interaction.to_csv('rules/human/omnipath.csv')
print(f'non-unique: {len(non_uniq)}')
print(non_uniq)
