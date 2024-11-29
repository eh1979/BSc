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
report_dir = sys.argv[5] # Fifth - Reports folder directory

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

######### Saving Processed Data #########

# Saves the background removed data to a .csv file with the same name as the original file name. Has two columns, theta and intensity

# --- Save Background Subtracted Original Data ---
save_filename = f"Sample_{sample_name}_XRR_data.csv"
save_filepath = os.path.join(report_dir, "XRR/Tables", save_filename)
pd.DataFrame({'theta': theta, 'intensity': intensity}).to_csv(save_filepath, index=False)
print(f"XRR data saved as {save_filepath}")


######### Plotting #########

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
