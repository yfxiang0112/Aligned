import numpy as np
import pandas as pd

# Load the NumPy array from the .npy file
npy_file_path = "dataset/prec1k/Y_train.npy"  # Replace with the path to your .npy file
array = np.load(npy_file_path)

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(array)

# Save the DataFrame to a .csv file
csv_file_path = "dataset/prec1k/Y_train.csv"  # Replace with the desired .csv file path
df.to_csv(csv_file_path, index=False)
