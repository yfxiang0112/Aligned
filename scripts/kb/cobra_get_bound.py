import numpy as np
from tqdm import tqdm
from pathlib import Path
import cobra
import pandas as pd
import json

def test_bound(model, reaction_id, ref_obj, reaction_2_id=None):
    reaction = model.reactions.get_by_id(reaction_id)
    rev = True if reaction.bounds[0] < 0 else False
    if reaction_2_id:
        reaction_2 = model.reactions.get_by_id(reaction_2_id)
        rev_2 = True if reaction_2.bounds[0] < 0 else False

    #bound_lst = [0, .03125, .0625,
    #              .125,.25,.375,.5,.625,.75,.875,
    #              1,2,3,4,5,6,7,8,9,10,20,45,50]
    bound_lst = [.5, .625, .75, .875,
                  1,2,3,4,5,6,7,8,9,10,20,45,50]
    for bound in bound_lst:
        reaction.bounds = (- bound, bound) if rev else (0,bound)
        if reaction_2_id:
            reaction_2.bounds = (- bound, bound) if rev_2 else (0,bound)
        solution = model.optimize()
        if abs(solution.objective_value - ref_obj) < 1e-3:
            reaction.bounds = (-1000,1000) if rev else (0,1000)
            if reaction_2_id:
                reaction_2.bounds = (-1000,1000) if rev_2 else (0,1000)
            return bound
    return 1000
    #else:
    #    reac_2 = model.reactions.get_by_id(reaction_2_id)
    #    for b1, b2 in zip(bound_lst, bound_lst):
    #        reaction.bounds = (- bound, bound) if rev else (0,bound)


gene_lst:pd.DataFrame = pd.read_csv('dataset/genes_241.csv', index_col=0)

# Load the iML1515 model
model = cobra.io.load_model('iML1515')

solution = model.optimize()
print("unconstrained growth rate:", solution.objective_value)
growth_wt = solution.objective_value

bound_dict = {}

''' test single reaction tight bound '''
for gene_id in tqdm(gene_lst['locus']):
    if gene_id not in model.genes:
        continue
    for reaction in model.genes.get_by_id(gene_id).reactions:
        if reaction.id in bound_dict:
            continue
        if reaction.bounds != (0,1000) and reaction.bounds != (-1000,1000):
            continue

        bound_dict[reaction.id] = test_bound(model, reaction.id, growth_wt)



#''' test 2-reaction tight bound '''
#reaction_lst = list(bound_dict.keys())
#for i, reac_1 in enumerate(tqdm(reaction_lst)):
#    for j, reac_2 in enumerate(reaction_lst[:i+1]):
#        if bound_dict[reac_1] > .5 or bound_dict[reac_2] > .5:
#            continue
#        bound = test_bound(model, reac_1, growth_wt, reaction_2_id=reac_2)
#        if bound_dict[reac_1] < bound:
#            bound_dict[reac_1] = bound
#        if bound_dict[reac_2] < bound:
#            bound_dict[reac_2] = bound
        

for reaction_id, bound in bound_dict.items():
    reaction = model.reactions.get_by_id(reaction_id)
    reaction.bounds = (0,bound) if reaction.bounds[0]==0 else (-bound,bound)

solution = model.optimize()
print("overall growth rate:", solution.objective_value)


with open('rules/fba_bound.json', 'w') as f:
    json.dump(bound_dict, f, indent=4)
