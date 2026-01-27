from cobra.io import load_model
import pandas as pd
import numpy as np
from scipy.sparse import coo, coo_matrix, save_npz
import time

# Load the iML1515 model
# First try to load from local models, if not found, load from BiGG database
try:
    model = load_model("iML1515")
except:
    print("Local model not found, downloading from BiGG database...")
    model = cobra.io.load_json_model("http://bigg.ucsd.edu/static/models/iML1515.json")

# Get the stoichiometric matrix (S matrix)
# Create a pandas DataFrame for better visualization
reaction_ids = [r.id for r in model.reactions]
metabolite_ids = [m.id for m in model.metabolites]

S_matrix = pd.DataFrame(
    data=0,
    index=metabolite_ids,
    columns=reaction_ids,
    dtype=float
)

# Fill in the stoichiometric coefficients
for reaction in model.reactions:
    for metabolite, coeff in reaction.metabolites.items():
        S_matrix.at[metabolite.id, reaction.id] = coeff

# Get the objective function
objective_reactions = [r.id for r in model.reactions if r.objective_coefficient != 0]
objective_coefficients = {r.id: r.objective_coefficient for r in model.reactions if r.objective_coefficient != 0}

# Print some information
print("Model loaded:", model.id)
print("Number of reactions:", len(model.reactions))
print("Number of metabolites:", len(model.metabolites))
print("\nObjective function:")
print(" + ".join([f"{coeff}*{rxn}" for rxn, coeff in objective_coefficients.items()]))

# Optionally save the S matrix to a CSV file
S_matrix.to_csv("rules/stoichiometric.csv")

# Show the first few rows and columns of the S matrix
print("\nPreview of the stoichiometric matrix:")
#objective = S_matrix.columns.isin(objective_coefficients).astype(np.float32)
objective = ['BIOMASS_Ec_iML1515_WT_75p37M']
#reindex = list(objective_coefficients.keys()) + [col for col in S_matrix.columns if col not in objective_coefficients.keys()]
reindex = list(objective) + [col for col in S_matrix.columns if col not in objective]
S_matrix = S_matrix[reindex]

'Restrict to only basic base metabolites '
#medium = pd.read_csv('rules/metabolites.csv')
medium = pd.read_csv('rules/full_supplement.csv')
S_matrix[(S_matrix.index.str.endswith('_e')) & ~(S_matrix.index.isin(medium['medium']))] = 0
#S_matrix = S_matrix.loc[~(S_matrix.index.isin(['h2o_c','h_c','h_p']))]
#S_matrix = S_matrix.loc[~(S_matrix.index.isin(medium['medium'])) & ~(S_matrix.index.isin(['h2o_c','h_c','h_p']))]
#                        ((S_matrix>0).sum(axis=1)>0) |
#                        ((abs(S_matrix[objective[0]])>.5))]
print(S_matrix)


gene_reaction_mapping = {}
for g in model.genes:
    gene_reaction_mapping[g.id] = list(np.nonzero(S_matrix.columns.isin([r.id for r in g.reactions]))[0])

S_matrix_np = S_matrix.to_numpy().astype(np.float32)

print(np.nonzero(objective))

S_consume = S_matrix_np<0
S_product = S_matrix_np>0

competitive_idx = np.nonzero((S_matrix.index=='atp_c') | (S_matrix.index=='h2o_c'))[0]# | (S_matrix.index.isin(medium.loc[medium['value']<100,'medium'])))[0]
#reactions = S_matrix.T @ S_matrix
adjacent_pos = (S_consume.T @ S_product).astype(int)
#adjacent_neg = (S_consume.T @ S_consume).astype(int)
adjacent_neg = (S_consume[competitive_idx,:].T @ S_consume[competitive_idx,:]).astype(int)
print(adjacent_neg.shape)

print(len(np.nonzero(adjacent_pos)[0]),len(np.nonzero(adjacent_neg)[0]))
print(adjacent_pos[:,0], adjacent_neg[:,0])
pos = set(np.nonzero(adjacent_pos[:,0])[0])
neg = set(np.nonzero(adjacent_neg[:,0])[0])
print(len(pos), len(neg))
print(len(pos-neg), len(neg-pos))

print(adjacent_pos)
save_npz('rules/gem_pos.npz', coo_matrix(adjacent_pos))
save_npz('rules/gem_neg.npz', coo_matrix(adjacent_pos))


label_set = pd.read_csv('dataset/label_set.csv', index_col=0)

gene_reac_annotation = np.zeros(shape=[len(label_set), adjacent_pos.shape[1]], dtype=bool)
for i,locus in enumerate(label_set['locus']):
    for j in gene_reaction_mapping[locus]:
        gene_reac_annotation[int(i), int(j)] = True

save_npz('rules/gem_annot.npz', coo_matrix(gene_reac_annotation))


#TODO
''' tmp: ensure annotation not change in subset selection '''
label_set = pd.read_csv('dataset/label_set_1096.csv', index_col=0)
gene_reac_annotation = np.zeros(shape=[len(label_set), adjacent_pos.shape[1]], dtype=bool)
for i,locus in enumerate(label_set['locus']):
    for j in gene_reaction_mapping[locus]:
        gene_reac_annotation[int(i),int(j)] = True

save_npz('rules/gem_annot_1096.npz', coo_matrix(gene_reac_annotation))

#t_0 = time.time()
#m = adjacent_neg @ adjacent_pos
#t = time.time() - t_0
#print('np mul time:',t)
#t_0 = time.time()
#m = adjacent_neg_pgb @ adjacent_pos_pgb
#t = time.time() - t_0
#print('pgb mul time:',t)
