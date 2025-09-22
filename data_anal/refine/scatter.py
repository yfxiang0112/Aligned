import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 1. Generate Sample Data with Groups ---
np.random.seed(42)
num_vectors = 10
num_dims = 300

reference_vector = np.random.randn(num_dims)
vectors = []

# Create two distinct groups with different characteristics
for i in range(num_vectors):
    if i < 5:  # Group 1: Smaller variance, positive bias
        noise_level = 0.3
        bias = 0.5
        group_color = '#1f77b4'  # Blue for Group A
    else:      # Group 2: Larger variance, negative bias
        noise_level = 0.8
        bias = -0.3
        group_color = '#ff7f0e'  # Orange for Group B
        
    v = reference_vector + bias + noise_level * np.random.randn(num_dims)
    vectors.append(v)
vectors = np.array(vectors)

# --- 2. Calculate Deviation ---

# Create group labels and vector names
groups = ['Group A'] * 5 + ['Group B'] * 5
vector_names = [f'G{A}_Vec {i+1}' for i, A in enumerate(['A']*5 + ['B']*5)]
group_colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 5

# --- 3. Create Publication-Quality Figure with Subplots ---
plt.style.use('seaborn-v0_8-white')
sns.set_palette("viridis")

# Create a 2x5 grid of subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

# Set consistent y-limits across all plots for fair comparison

# --- 4. Create Scatter Plot for Each Vector ---
for i, (ax, vec_name, group_color, deviation_data) in enumerate(zip(axes, vector_names, group_colors, deviation_matrix)):

    y_min, y_max = np.percentile(all_deviations, [1, 99])  # Use 1st/99th percentiles to exclude extreme outliers
    y_buffer = (y_max - y_min) * 0.1  # Add 10% buffer
    y_lim = (y_min - y_buffer, y_max + y_buffer)
    
    # Create scatter plot
    x_values = np.arange(1, num_dims + 1)  # Dimension indices: 1 to 300
    scatter = ax.scatter(x_values, deviation_data, 
                        alpha=0.6, s=15,  # Smaller points for density
                        color=group_color, 
                        edgecolors='none')
    
    # Add reference line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.8)
    
    # Add mean deviation line
    mean_deviation = np.mean(deviation_data)
    ax.axhline(y=mean_deviation, color='black', linestyle='-', linewidth=1.5, alpha=0.9,
              label=f'Mean: {mean_deviation:.2f}')
    
    # Set consistent limits and labels
    ax.set_ylim(y_lim)
    ax.set_xlim(0, num_dims + 1)
    ax.set_title(f'{vec_name}\n', fontsize=11, fontweight='bold', fontfamily='sans-serif')
    
    # Add grid for readability
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    
    # Add legend for mean line
    ax.legend(loc='upper right', fontsize=9, frameon=True)
    
    # Only add y-label to leftmost plots
    if i % 5 == 0:
        ax.set_ylabel('Deviation from Reference', fontsize=10, fontfamily='sans-serif')
    
    # Add x-label to bottom plots
    if i >= 5:
        ax.set_xlabel('Dimension Index', fontsize=10, fontfamily='sans-serif')
    else:
        ax.tick_params(labelbottom=False)  # Hide x labels for top row

# --- 5. Add Group Labels and Divider ---
# Add group labels
fig.text(0.25, 0.95, 'Group A', ha='center', va='center', 
         fontsize=12, fontweight='bold', fontfamily='sans-serif', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='#1f77b4', alpha=0.2))
fig.text(0.75, 0.95, 'Group B', ha='center', va='center', 
         fontsize=12, fontweight='bold', fontfamily='sans-serif',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='#ff7f0e', alpha=0.2))

# Add vertical divider line between groups
fig.add_artist(plt.Line2D([0.5, 0.5], [0.1, 0.9], transform=fig.transFigure, 
                         color='black', linestyle='--', linewidth=2, alpha=0.7))

# Add overall title
plt.suptitle('Deviation from Reference Vector Across 300 Dimensions', 
             fontsize=16, fontweight='bold', fontfamily='sans-serif', y=0.98)

# --- 6. Final Layout and Save ---
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.4)  # Adjust spacing

plt.savefig('scatter_grid_vector_deviation.png', 
            dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
