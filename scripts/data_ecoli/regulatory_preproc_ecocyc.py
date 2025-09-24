import pandas as pd

file_path = 'rules/ecoli/regulatory.txt'

# Initialize counters
pos_count = 0
neg_count = 0
dual_count = 0

regu_pos_count = 0
regu_neg_count = 0
regu_dual_count = 0

regulator = []
regulated = []
edge = []


with open(file_path, 'r') as file:
    lines = file.readlines()

# Iterate through the lines of the file
i = 0
while i < len(lines):
    # The first line is the outgoing node
    out_node = lines[i].strip()
    #regulators.add(out_node)
    i += 1

    # The second line contains all adjacent nodes
    neighbors = lines[i].strip().split()
    i += 1

    # Iterate through the adjacent nodes
    for neighbor in neighbors:
        #if not neighbor.endswith('*'):
        #    regulated.add(neighbor)

        # Check the type of edge
        regulator.append(out_node[:-1])

        if neighbor.startswith('+/-'):
            edge.append('-+')
            if neighbor.endswith('*'):
                regu_dual_count += 1
                regulated.append(neighbor[3:-1])
            else:
                dual_count += 1
                regulated.append(neighbor[3:])


        elif neighbor.startswith('+'):
            edge.append('+')
            if neighbor.endswith('*'):
                regu_pos_count += 1
                regulated.append(neighbor[1:-1])
            else:
                pos_count += 1
                regulated.append(neighbor[1:])

        elif neighbor.startswith('-'):
            edge.append('-')
            if neighbor.endswith('*'):
                regu_neg_count += 1
                regulated.append(neighbor[1:-1])
            else:
                neg_count += 1
                regulated.append(neighbor[1:])

        else:
            print(neighbor)
            regulator.pop(-1)

    #if len(regulated) < len(regulator):
    #    print(out_node)
    #    print(neighbors)
    #    exit()

print(len(regulator), len(regulated), len(edge))
df = pd.DataFrame({'regulator':regulator, 'edge':edge, 'regulated':regulated})
df.to_csv('rules/ecoli/regulatory_ecocyc.csv', index=False)
print(df)

# Output the results
print(f"{len(set(regulator))} regulator genes, {len(set(regulated))} regulated genes")
print(f"regulated '+': {pos_count}")
print(f"regulated '-': {neg_count}")
print(f"regulated '+/-': {dual_count}")

print(f"regulator '+': {regu_pos_count}")
print(f"regulator '-': {regu_neg_count}")
print(f"regulator '+/-': {regu_dual_count}")
