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


###### Ag and MnN Peak Position #######
silver_peaks = {
    'Ag 111': 38.06,
    'Ag 200': 44.24,
    'Ag 220': 64.35,
    'Ag 311': 77.28,
    'Ag 222': 81.41,
}

MnN_peaks = {
    'MnN 111': 37.12,
    'MnN 200': 42.91,
    'MnN 002': 43.57,
    'MnN 220': 62.2,
    'MnN 022': 62.9,
    'MnN 310': 70.67,
    'MnN 222': 79.08,
}

# Create bulk_peaks dictionary and add both to it.  If I change material later can just add a new set of peaks and then add it to the bulk peaks without having to change the code further down.
bulk_peaks = {}
bulk_peaks.update(silver_peaks)
bulk_peaks.update(MnN_peaks)

# Now the bulk_peaks dictionary contains data from both

##### Access Bash Script File Path: #####
directory = sys.argv[1]  # The first argument - Data folder
graphs_dir = sys.argv[2] # Second - graphs folder
table_dir = sys.argv[3] # Third - Values folder
sample_name = sys.argv[4] # Fourth - sample name
report_dir = sys.argv[5] # Fifth - Reports folder directory

tab_filepath= os.path.join(report_dir, "2thetachi/Tables") # Defines a table filepath within the report directory

#######   Function Initialisation  ######

def extract_data_from_file(file_path):
    def is_comment(line):
        return line.startswith('#')

    with open(file_path, 'r') as f:
        skip_rows = 1

    data = pd.read_csv(file_path, sep=',', header=None, names=['theta', 'intensity'])
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    theta = data['theta'].values
    original_intensity_corrected = data['intensity'].values

    return theta, original_intensity_corrected

##########  Plot 1: ############

## Overlay Plot of the Background subtracted but unbinned data for all 2thetachi samples

# --- Plot 3: Background-Subtracted Original Unbinned Data and Gaussians ---
plt.figure(figsize=(12, 6))

# Labels and title
plt.xlabel('2θ (Degrees)')
plt.ylabel('Intensity (Counts)')
#plt.title('Background Subtracted Original Unbinned Data with Fitted Gaussians')

#cmap = cm.get_cmap('Set1')  # chosen colormap
cmap = mcolors.ListedColormap(["magenta","black","blue","darkblue","fuchsia","indigo"])
colours = [cmap(i) for i in range(cmap.N)]  # Extract all colours from the colormap

# Create a colour iterator to cycle through
colour_cycle = iter(colours)

for filename in os.listdir(tab_filepath):
    if filename.endswith("background_subtracted_data.csv"):
        file_path = os.path.join(tab_filepath, filename)

        Colour = next(colour_cycle)

        #Select just the sample name from the filename
        sample_name_label = re.search(r"Sample_(.*)_background", filename)
        label_from_file=sample_name_label.group(1) # Group 1 is the (.*) section

        # call function which returns theta and original_intensity_corrected
        theta, original_intensity_corrected = extract_data_from_file(file_path)

        # Corrected original unbinned intensity after background subtraction
        plt.plot(theta, original_intensity_corrected, label=label_from_file, color=Colour, lw=1)

for label, theta in bulk_peaks.items():
    # Plot the bulk line
    plt.axvline(x=theta, linestyle='-', color='black', linewidth=1.5, zorder=2)

    # Get the current y-axis limitsK
    ymin, ymax = plt.ylim()

    # Place the label based on y-axis limits, near the top of the plot. Can be adjusted up and down.
    label_position = ymax * 1.015  #  Adjust if needed; 1 = top, 0 = bottom
    plt.text(theta, label_position, label, color='black', rotation=90,
             verticalalignment='bottom', horizontalalignment='center', fontsize = 7)

plt.legend()
plt.grid(True)

# Save the plot of background-subtracted original unbinned data with Gaussians
fin_filename = '2thetachi_overlay.png'
fin_filepath = os.path.join(report_dir, "2thetachi/Graphs", fin_filename)
plt.savefig(fin_filepath)
print("Plot of overlay 2thetachi data saved as '2thetachi_overlay.png'")
