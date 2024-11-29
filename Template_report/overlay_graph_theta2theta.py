import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from brokenaxes import brokenaxes
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

tab_filepath= os.path.join(report_dir, "2theta/Tables") # Defines a table filepath within the report directory

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

################# Plot 1 ###########################

##########  Graph Plotting #############



##### ColourMap #####

#cmap = cm.get_cmap('Set1')  # chosen colormap
cmap = mcolors.ListedColormap(["lime","darkgreen","blue","darkblue","fuchsia","indigo"])
colours = [cmap(i) for i in range(cmap.N)]  # Extract all colours from the colormap

# Create a colour iterator to cycle through
colour_cycle = iter(colours)

##### Plot Graph: #####

# Plot the data with brokenaxis
fig = plt.figure(figsize=(12, 6))
#brk = brokenaxes(xlims=((theta.min(), peak_2_min), (peak_2_max, peak_1_min), (peak_1_max, theta.max())), wspace=0.05)

# Set log y-axis scale
#for ax in brk.axs:
    #ax.set_yscale('log')

# Initialise brokenaxes with appropriate x-axis breaks
brk = brokenaxes( xlims=((30, 32), (33, 65.5), (79.5, 85)), wspace=0.05 )


for filename in os.listdir(tab_filepath):
    if filename.endswith("background_subtracted_data.csv"):
        file_path = os.path.join(tab_filepath, filename)

        Colour = next(colour_cycle)

        #Select just the sample name from the filename
        sample_name_label = re.search(r"Sample_(.*)_background", filename)
        label_from_file=sample_name_label.group(1) # Group 1 is the (.*) section


         # call function which returns theta and original_intensity_corrected
        theta, original_intensity_corrected = extract_data_from_file(file_path)


        ##### Identify and Remove Si Peaks ####

        # Identify the silicon peak at 69 degrees
        silicon_peak_index_1 = np.argmax(original_intensity_corrected)
        silicon_peak_theta_1 = theta[silicon_peak_index_1]
        silicon_peak_intensity = original_intensity_corrected[silicon_peak_index_1]

        # For the second peak, define the theta range for the search
        theta_min, theta_max = 30, 35
        in_range = (theta >= theta_min) & (theta <= theta_max)

        # Identify the silicon peak at 33 degrees
        silicon_peak_index_2 = np.argmax(original_intensity_corrected[in_range])
        silicon_peak_theta_2 = theta[silicon_peak_index_2]
        silicon_peak_intensity = original_intensity_corrected[silicon_peak_index_2]

        # Set the location and ranges of the peaks for masking and for the graph break.
        peak_1_min = 65.5
        peak_1_max = 79.5

        peak_2_range = 0.2
        peak_2_min = silicon_peak_theta_2 - peak_2_range
        peak_2_max = silicon_peak_theta_2 + peak_2_range

        # Remove the silicon peak by setting values to NaN in a small range around the peak.  This allows for the plot to dynamically adjust the scaling of the y axis so the data is as large as possible.
        mask_1 = (theta >= peak_1_min) & (theta <= peak_1_max)
        original_intensity_corrected[mask_1] = np.nan

        mask_2 = (theta >= peak_2_min) & (theta <= peak_2_max)
        original_intensity_corrected[mask_2] = np.nan

        #brk = brokenaxes(xlims=((theta.min(), peak_2_min), (peak_2_max, peak_1_min), (peak_1_max, theta.max())), wspace=0.05)

        # Corrected original unbinned intensity after background subtraction
        brk.plot(theta, original_intensity_corrected, label=label_from_file, color=Colour, lw=1)


#Plot the bulk peak lines
for label, theta in bulk_peaks.items():
    # Plot the bulk line
    brk.axvline(x=theta, linestyle='-', color='black', linewidth=1.5, zorder=2)

    # Determine which subplot the peak belongs to
    for ax in brk.axs:
        xlim = ax.get_xlim()
        if xlim[0] <= theta <= xlim[1]:
            # Get the current y-axis limits for the subplot
            ymin, ymax = ax.get_ylim()
            label_position = ymax * 1.05  # Adjust label position as needed

            # Add the label within the corresponding subplot
            ax.text(
                theta, label_position, label, color='black', rotation=90,
                verticalalignment='bottom', horizontalalignment='center', fontsize=7
            )
            break  # Label placed, no need to check other subplots


# Plot lines where the original theta peaks were.
brk.axvline(32.95, color='r', linestyle='', label=f'Silicon Peak at 32.95°')
brk.axvline(69.13, color='r', linestyle='', label=f'Silicon Peak at 69.13°')


# Set parameters for Graph
brk.set_xlabel('2θ (Degrees)')
brk.set_ylabel('Intensity (Counts)')
#brk.set_title('θ-2θ Plot with Silicon Peak Removed')
brk.legend()
brk.grid(True)


# Save the plot of background-subtracted original unbinned data with Gaussians
fin_filename = 'theta2theta_overlay.png'
fin_filepath = os.path.join(report_dir, "theta2theta/Graphs", fin_filename)
plt.savefig(fin_filepath)
print("Plot of overlay theta2theta data saved as 'theta2theta_overlay.png'")
