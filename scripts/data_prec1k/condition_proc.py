import pandas as pd
import re
import numpy as np

from chem2vec import get_smiles, generate_ecfp

def extr_chem(s):
    ''' extract chemical name '''
    if type(s) != str:
        return 'nothing'
    res = re.findall(r'[\d\w-]+', s)
    if len(res) == 0:
        return None
    else:
        return res[0]

def extr_conc(s):
    ''' extract concentration & process unit '''
    if type(s) != str:
        return 0

    res = re.findall(r'\([\d\w/\.\%]+\)', s)
    res = res[0][1:-1] if len(res)>0 else '1'
    conc = re.findall(r'[\d\.]+', res)[0]
    conc = float(conc)
    #if conc == 0:
    #    print(s)
    #    exit()
    unit = re.findall(r'[a-zA-Z/]+', res)
    unit =  unit[0] if len(unit)>0 else None

    match unit:
        case None:
            pass
        case 'ug/mL':
            conc /= 1000
        case 'mg/L':
            conc /= 1000
        case 'mg/mL':
            pass
            #TODO: mol concentration convert
        case 'v/w%':
            pass
            #TODO
        case 'uM':
            conc /= 1000
        case 'mM':
            pass
        case 'M':
            conc *= 1000
        case _:
            print(unit)
            #pass
    return conc


if __name__ == '__main__':
    
    ''' labeled data processing '''
    #metadata = pd.read_csv('dataset/metadata.csv')
    #''' define the converted df '''
    #data_in = {
    #        'temperature':     metadata['Temperature (C)'],
    #        'pH':              metadata['pH'],
    #        'carbon_src':      metadata['Carbon Source (g/L)'].apply(extr_chem),
    #        'C_carbon_src':    metadata['Carbon Source (g/L)'].apply(extr_conc),
    #        'nitrogen_src':    metadata['Nitrogen Source (g/L)'].apply(extr_chem),
    #        'C_nitrogen_src':  metadata['Nitrogen Source (g/L)'].apply(extr_conc),
    #        'supplement':      metadata['Supplement'].apply(extr_chem),
    #        'C_supplement':    metadata['Supplement'].apply(extr_conc),
    #        'growth_rate':     metadata['Growth Rate (1/hr)'].fillna(0)
    #        }
    #data_in = pd.DataFrame(data_in)
    #print(data_in)
    #data_in.to_csv('dataset/metadata_sel.csv')
    
    
    ''' unlabeled data processing '''
    data_in = pd.read_csv('dataset/metadata_sel_unlabels.csv')
    
    #####
    
    ''' construct mapping '''
    csrc_set = set(extr_chem(s) for s in data_in['carbon_src'].unique())
    nsrc_set = set(extr_chem(s) for s in data_in['nitrogen_src'].unique())
    suppl_set = set(extr_chem(s) for s in data_in['supplement'].unique())
    chem_set = set.union(csrc_set, nsrc_set, suppl_set)
    smiles_mapping = {c : get_smiles(c) for c in chem_set}
    
    ''' convert to SMILES and generate ECFP vectors of chemicals '''
    csrc_vec = generate_ecfp(data_in['carbon_src'].apply(lambda x: smiles_mapping[x]))
    nsrc_vec = generate_ecfp(data_in['nitrogen_src'].apply(lambda x: smiles_mapping[x]))
    suppl_vec = generate_ecfp(data_in['supplement'].apply(lambda x: smiles_mapping[x]))
    
    ''' concate into input matrix '''
    X = np.hstack((data_in[['temperature','pH']], csrc_vec, data_in[['C_carbon_src']], nsrc_vec, data_in[['C_nitrogen_src']], suppl_vec, data_in[['C_supplement','growth_rate']]))
    print(X)
    print(X.shape)
    #np.save('dataset/X_train.npy', X)
    np.save('dataset/X_unlabel.npy', X)
