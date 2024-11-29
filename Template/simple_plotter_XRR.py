import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#########    File Path:    ############
# Change to needed file
#file_path = 'EOH1_XRR.txt'

directory = sys.argv[1]  # The first argument - (Data folder)
print(directory)
graphs_dir = sys.argv[2] # Second - graphs folder
table_dir = sys.argv[3] # Third - Values folder
sample_name = sys.argv[4] # Fourth - sample name (used this primarily in VSM to print out sample name just to check I'd inputted the right one.

def extract_data_from_file(file_path):
    def is_comment(line):
        return line.startswith('#')

    with open(file_path, 'r') as f:
        skip_rows = sum(1 for line in f if is_comment(line))

    data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=skip_rows, names=['theta', 'intensity'])
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    theta = data['theta'].values
    intensity = data['intensity'].values

    return theta, intensity

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)

# Extract data from the file
theta, intensity = extract_data_from_file(file_path)

plt.figure(figsize=(12, 6))

plt.plot(theta, intensity, label='Data', color='b', lw=1)

# Labels and title
plt.yscale('log')
plt.xlabel('???')
plt.ylabel('Intensity (Counts)')
plt.title('Reflectivity Plot')
plt.legend()
plt.grid(True)

filename = 'reflectivity_plot.png'
output_path = os.path.join(graphs_dir, filename)
plt.savefig(output_path)
print("Plot saved as 'reflectivity_plot.png'")
