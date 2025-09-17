import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate base data
df = pd.read_csv('data_anal/refine/log.csv', index_col=0)
x = ['Original\nKB', 'Baseline\n0%', '5%', '10%', '20%', '30%', '40%', '50%', '70%', '90%']
y_mod = df['modularity_mean']
e_mod = df['modularity_tol']
y_aso = df['degree_assortativity_mean']
e_aso = df['degree_assortativity_tol']

plt.figure(figsize=(12, 7))

# Plot smooth line
plt.plot(x, y_mod, 'b-', linewidth=2, label='Modularity (Ascend)', alpha=0.7)
plt.plot(x, y_aso, 'g-', linewidth=2, label='Assortativity (Descend)', alpha=0.7)

# Plot points with variations
plt.errorbar(x, y_mod, yerr=e_mod, fmt='o', color='blue', 
             markersize=8, capsize=5, capthick=2, alpha=0.8)

plt.errorbar(x, y_aso, yerr=e_aso, fmt='o', color='green', 
             markersize=8, capsize=5, capthick=2, alpha=0.8)

plt.legend(fontsize=12)
plt.xlabel('Removed Arcs', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Structural Scores of Reconstructed GRNs with Different Removed Arc Portions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_anal/refine/structural.png', dpi=300)
plt.show()

y_reg = df['f1_initial_KB_combined_mean']
e_reg = df['f1_initial_KB_combined_tol']
y_clo = df['f1_closure_KB_combined_mean']
e_clo = df['f1_closure_KB_combined_tol']

plt.figure(figsize=(12, 7))

# Plot smooth line
plt.plot(x, y_reg, 'b-', linewidth=2, label='F1 of Reconstructed GRN', alpha=0.7)
plt.plot(x, y_clo, 'g-', linewidth=2, label='F1 of Reconstructed R^(k)', alpha=0.7)

# Plot points with variations
plt.errorbar(x, y_reg, yerr=e_reg, fmt='o', color='blue', 
             markersize=8, capsize=5, capthick=2, alpha=0.8)

plt.errorbar(x, y_clo, yerr=e_clo, fmt='o', color='green', 
             markersize=8, capsize=5, capthick=2, alpha=0.8)

plt.legend(fontsize=12)
plt.xlabel('Removed Arcs', fontsize=12)
plt.ylabel('Reconstructed F1 Scores', fontsize=12)
plt.title('Reconstruction Accuracy of Reconstructed GRNs with Different Removed Arc Portions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_anal/refine/reconstruction_f1.png', dpi=300)
plt.show()
