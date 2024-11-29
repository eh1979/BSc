import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv
import os
import sys
import re
import random
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors

##### Access Bash Script File Path: #####
directory = sys.argv[1]  # The first argument - Data folder
graphs_dir = sys.argv[2] # Second - graphs folder
table_dir = sys.argv[3] # Third - Values folder
sample_name = sys.argv[4] # Fourth - sample name
report_dir = sys.argv[5] # Fifth - Reports folder directory

tab_filepath= os.path.join(report_dir, "XRR/Tables") # Defines a table filepath within the report directory

#######   Function Initialisation  ######

def extract_data_from_file(file_path):
    data = pd.read_csv(file_path, sep=',', header=0)
    data = data.apply(pd.to_numeric, errors='coerce')
    #data.dropna(inplace=True)

    theta = data['theta'].values
    intensity = data['intensity'].values

    return theta, intensity


##########  Plot 1: ############

## Overlay Plot of the Background subtracted but unbinned data for all 2thetachi samples

# --- Plot 3: Background-Subtracted Original Unbinned Data and Gaussians ---
plt.figure(figsize=(12, 6))

# Labels and title
plt.yscale('log')
plt.xlabel('2θ (Degrees)')
plt.ylabel('Intensity (Counts)')
#plt.title('Reflectivity Plot')

#cmap = cm.get_cmap('Set1')  # chosen colormap
cmap = mcolors.ListedColormap(["lime","darkgreen","blue","darkblue","fuchsia","indigo"])
colours = [cmap(i) for i in range(cmap.N)]  # Extract all colours from the colormap

# Create a colour iterator to cycle through
colour_cycle = iter(colours)

for filename in os.listdir(tab_filepath):
    if filename.endswith("XRR_data.csv"):
        file_path = os.path.join(tab_filepath, filename)

        Colour = next(colour_cycle)

        #Select just the sample name from the filename
        sample_name_label = re.search(r"Sample_(.*)_XRR", filename)
        label_from_file=sample_name_label.group(1) # Group 1 is the (.*) section

        # call function which returns theta and original_intensity_corrected
        theta, intensity = extract_data_from_file(file_path)

        # Corrected original unbinned intensity after background subtraction
        plt.plot(theta, intensity, label=label_from_file, color=Colour, lw=1)


plt.legend()
plt.grid(True)

# Save the plot of background-subtracted original unbinned data with Gaussians
fin_filename = 'XRR_overlay.png'
fin_filepath = os.path.join(report_dir, "XRR/Graphs", fin_filename)
plt.savefig(fin_filepath)
print("Plot of overlay XRR data saved as 'XRR_overlay.png'")
