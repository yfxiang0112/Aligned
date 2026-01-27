import numpy as np
from pathlib import Path
import cobra
import pandas as pd
import json


gene_lst:pd.DataFrame = pd.read_csv('dataset/genes_241.csv', index_col=0)

# Load the iML1515 model
model = cobra.io.load_model('iML1515')

solution = model.optimize()
print("Objective value (growth rate):", solution.objective_value)
growth_wt = solution.objective_value


with open('rules/fba_bound.json', 'r') as f:
    bound_dict = json.load(f)


for reaction_id, bound in bound_dict.items():
    reaction = model.reactions.get_by_id(reaction_id)
    reaction.bounds = (0,bound/2) if reaction.bounds[0]==0 else (-bound/2,bound/2)

solution = model.optimize()
print("Objective value (growth rate):", solution.objective_value)
ref_growth = solution.objective_value


for gene_id in gene_lst['locus']:
    #for gene_2 in gene_lst['locus']:
    if gene_id not in model.genes:
        continue
    reactions = model.genes.get_by_id(gene_id).reactions

    #for reaction in model.genes.get_by_id(gene_id).reactions:
    #    #print(reaction.bounds)
    #    if reaction.id in bound_dict:
    #        continue
    #    if reaction.bounds == (0,1000):
    #        bound_dict[reaction.id] = test_bound(model, reaction.id, rev=False, ref_obj=growth_wt)
    #    elif reaction.bounds == (-1000,1000):
    #        bound_dict[reaction.id] = test_bound(model, reaction.id, rev=True, ref_obj=growth_wt)

    for reaction in reactions:
        reaction.bounds = (reaction.bounds[0]*10, reaction.bounds[1]*10)

    solution = model.optimize()
    print(f"{gene_id} upregulated growth rate:{solution.objective_value}")

    for reaction in reactions:
        reaction.bounds = (reaction.bounds[0]/10, reaction.bounds[1]/10)

    #solution = model.optimize()
    #print(f"{gene_id} upregulated growth rate:{solution.objective_value}")

    #for reaction in reactions:
    #    reaction.bounds = (reaction.bounds[0]/2, reaction.bounds[1]/2)

    #print("\nFluxes for the overexpressed reaction(s):")
    #for reaction in reactions:
    #    print(f"{reaction.id}: {solution.fluxes[reaction.id]}")
    #
    #print("\nExchange reaction fluxes:")
    #for reaction in model.exchanges:
    #    print(f"{reaction.id}: {solution.fluxes[reaction.id]}")
