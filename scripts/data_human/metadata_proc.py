import pickle
import numpy as np
import pandas as pd

#NOTE tmp
data_norman = pickle.load(open('../GEARS/norman/data_pyg/cell_graphs.pkl','rb'))
data_dixit = pickle.load(open('../GEARS/dixit/data_pyg/cell_graphs.pkl','rb'))
data_adamson = pickle.load(open('../GEARS/adamson/data_pyg/cell_graphs.pkl','rb'))

keys = set(data_norman.keys()).union(set(data_dixit.keys())).union(set(data_adamson.keys()))
print(len(data_norman.keys()))
print(len(keys))

data = [(data_norman[k] if k in data_norman else []) + 
        (data_dixit[k] if k in data_dixit else [])+ 
        (data_adamson[k] if k in data_adamson else [])
        for k in keys]

Y = []
metadata = []
for i, lst in enumerate(data):
    pert_idx = lst[0].pert_idx
    pert = lst[0].pert
    de_idx = lst[0].de_idx

    start_data_idx = len(Y)
    Y += [(np.squeeze(d.y, 0) - np.squeeze(d.x, -1))[:5012] for d in lst]
    end_data_idx = len(Y)

    metadata.append({'pert':pert, 'pert_idx':pert_idx, 'data_start_idx':start_data_idx, 'data_end_idx+1':end_data_idx, 'de_idx':de_idx})

Y = np.stack(Y)
print(len(Y))
print(Y)
np.save('dataset/human/Y.npy', Y)

metadata = pd.DataFrame(metadata)
print(metadata)
metadata.to_csv('dataset/human/metadata.csv')
