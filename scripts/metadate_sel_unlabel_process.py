import pandas as pd
import numpy as np

df = pd.read_csv('dataset/metadata_sel.csv')

columns = []
for column in df.iloc[:, 1:].columns:
    unique_values = df[column].unique()
    columns.append(unique_values)

add_carbon_scr = ['mannital','lactose','malate','citrate','ribulose']
add_n_scr = ['urea','alanine','asparagine','arginine','histidine']
add_supplement = ['vitamin b12', 'magnesium sulfate', 'calcium chloride', 'nicotinamide', 'blotin']

columns[2] = np.append(columns[2], add_carbon_scr) 
columns[4] = np.append(columns[4], add_n_scr)      
columns[6] = np.append(columns[6], add_supplement)  

new_data = {}
num_new_rows = 5000

new_data['index'] = range(num_new_rows)

for i, column in enumerate(df.iloc[:, 1:].columns):
    new_data[column] = np.random.choice(columns[i], size=num_new_rows)

new_df = pd.DataFrame(new_data)

cols = new_df.columns.tolist()
cols.insert(0, cols.pop(cols.index('index')))
new_df = new_df.reindex(columns=cols)
new_df.to_csv('dataset/metadata_sel_unlabels.csv', index=False)