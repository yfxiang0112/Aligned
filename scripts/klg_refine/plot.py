import re
import numpy as np
import matplotlib.pyplot as plt

# Read the log file
with open("scripts/klg_refine/log_exp_lr_decay_t_increase.txt", "r") as f:
    log_data = f.read()

# Regex pattern to extract iteration and losses
pattern = re.compile(
    r"Iteration (\d+): Loss = ([\d.]+)\s*"  
    r"\|Xk-Y\|_F:\s*([\d.]+),\s*\|X-X0\|:\s*([\d.]+)\s*"  
    r"rounded \|X_k-Y\|_0 =\s*([\d.]+), f1 =\s*([\d.]+)\s*, approx slack:\s*([\d.]+)\s*"
    r"rounded before pow \|X_k-Y\|_0 =\s*([\d.]+), f1 =\s*([\d.]+)\s*"
)

# Extract all matches
matches = pattern.findall(log_data)

# Convert to structured data
iterations = []
total_loss = []
loss1 = []
loss2 = []
#acc = []
f1 = []

for match in matches:
    iterations.append(int(match[0]))
    total_loss.append(float(match[1]))
    loss1.append(float(match[2]))
    loss2.append(float(match[3]))
    #acc.append(1- int(match[4])/(4639**2))
    f1.append(float(match[8]))

iterations = iterations[:50]
total_loss, loss1, loss2, f1 = total_loss[:50], loss1[:50], loss2[:50], f1[:50]

# Plotting
#plt.figure(figsize=(10, 6))
#
#plt.plot(iterations, total_loss, label="Total Loss", marker='o', linestyle='-', linewidth=2)
#plt.plot(iterations, loss1, label="|Xk-Y|_F", marker='x', linestyle='--')
#plt.plot(iterations, loss2, label="|X-X0|_1", marker='^', linestyle='-.')
#plt.plot(iterations, loss3, label="Dataset Acc", marker='s', linestyle=':')
#
#plt.xlabel("Iteration")
#plt.ylabel("Loss Value")
#plt.title("Training Loss Over Iterations")
#plt.legend()
#plt.grid(True)


fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Total Loss (leftmost y-axis)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (*1e6)", color='tab:blue')
#ax1.set_ylim(0, 1e6)
ax1.plot(iterations, total_loss, label="Total Loss", color='tab:blue', marker='o', linestyle='-', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a 2nd y-axis (f1)
ax2 = ax1.twinx()
ax2.spines['right'].set_position(('outward', 0))
ax2.set_ylabel("F1", color='tab:red')
ax2.plot(iterations, f1, label="F1 on Train Dataset", color='tab:red', marker='s', linestyle=':')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Create a 3rd y-axis (Loss 1)
#ax3 = ax1.twinx()
#ax3.spines['right'].set_position(('outward', 60))  # Shift axis to prevent overlap
#ax3.set_ylabel("MSE", color='tab:orange')
#ax3.plot(iterations, loss1, label="MSE loss: |Xk-Y|_F^2", color='tab:orange', marker='x', linestyle='--')
#ax3.tick_params(axis='y', labelcolor='tab:orange')
ax1.plot(iterations, loss1, label="MSE Loss on Dataset: |Xk-Y|_F^2", color='tab:orange', marker='x', linestyle='--')

# Create a 4th y-axis (Loss 2)
#ax4 = ax1.twinx()
#ax4.spines['right'].set_position(('outward', 120))
#ax4.set_ylabel("Regularization", color='tab:green')
#ax4.plot(iterations, loss2, label="Regularization: |X-X0|_1", color='tab:green', marker='^', linestyle='-.')
#ax4.tick_params(axis='y', labelcolor='tab:green')
ax1.plot(iterations, loss2, label="Sparse Regularization: |X-X0|_1", color='tab:green', marker='^', linestyle='-.')

# Combine legends from all axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
#lines3, labels3 = ax3.get_legend_handles_labels()
#lines4, labels4 = ax4.get_legend_handles_labels()
#ax1.legend(lines1 + lines2 + lines3 + lines4, labels1 + labels2 + labels3 + labels4, loc='lower right')
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.title("Loss and F1 on Precise Dataset of Sparse Knowledge Refinement")
plt.grid(False)  # Disable grid to avoid confusion
plt.tight_layout()
plt.savefig('scripts/klg_refine/loss.png', dpi=300)
plt.show()
