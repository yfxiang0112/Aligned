from operator import index
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 1. Generate Sample Data ---
gsr_result = pd.read_csv('data_anal/refine/gsr_mix.csv', index_col=['p_incomp', 'score_type'])
reference_vector = np.array(gsr_result.loc[('orig','mean_auprc_w')])

# Create a set of 10 vectors 'V' that vary from 'R' in different ways
vectors = gsr_result.loc[[x for x in gsr_result.index if x[1]=='mean_auprc_w']]
num_vectors = len(vectors)


vectors = np.array(vectors) # Shape: (10, 300)

# --- 2. Calculate Deviation from Reference ---
# This creates a matrix where each row is the deviation of one vector from R
deviation_matrix = vectors - reference_vector # Shape: (10, 300)

# --- 3. Prepare Data for Plotting (Long Format) ---
# We need a DataFrame with: Vector ID, and all 300 deviation values for that vector
data_for_plot = []
vector_names = [f'Vec {i+1}' for i in range(num_vectors)] # Create descriptive names

for vec_index, vec_name in enumerate(vector_names):
    for dim_value in deviation_matrix[vec_index]:
        data_for_plot.append({'Vector': vec_name, 'Deviation': dim_value})

df = pd.DataFrame(data_for_plot)

# --- 4. Create a Publication-Quality Figure with Subplot ---
# Set the visual style for a formal publication
plt.style.use('seaborn-v0_8-white') # Clean, minimal style
sns.set_palette("viridis") # Use a colorblind-friendly palette

# Create the figure and the specific subplot we want to use
fig, ax = plt.subplots(1, 1, figsize=(10, 6)) # 1 subplot, figsize in inches
# figsize is (width, height). Adjust to fit your publication's column width.

# --- 5. Create the Violin Plot on the Axes ---
# Plotting on our specific subplot 'ax'
sns.violinplot(data=df, x='Vector', y='Deviation', ax=ax, cut=0, inner='box') 
# 'cut=0': limits the violin shape to the data range
# 'inner='box'': adds a boxplot inside for quartiles and median

# --- 6. Enhance the Plot for Publication ---
# Add a critical reference line at y=0
ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.8, label='Reference Vector')

# Improve labels and title (use sans-serif fonts common in publications)
ax.set_xlabel('Vector', fontsize=12, fontfamily='sans-serif')
ax.set_ylabel('Deviation from Reference', fontsize=12, fontfamily='sans-serif')
ax.set_title('Distribution of Feature Deviations for Each Vector', 
             fontsize=14, fontweight='bold', fontfamily='sans-serif')

# Improve tick labels for readability
ax.tick_params(axis='both', which='major', labelsize=10)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right') # Rotate labels if long

# Add a grid for easier reading of values, make it subtle
ax.grid(axis='y', linestyle=':', alpha=0.4)

# Optional: Add a legend for the reference line
ax.legend(loc='upper right', frameon=True)

# --- 7. Final Layout Adjustments ---
# Tighten the layout to prevent clipping of labels
plt.tight_layout()

# --- 8. Save in High Resolution for Publication ---
plt.savefig('vector_deviation_violin_plot.png', 
            dpi=600, # High resolution for printing
            bbox_inches='tight', # Ensures no labels are cut off
            facecolor='white', # White background, not transparent
            edgecolor='none') 

plt.show() # Display the plot
