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

# Create bulk_peaks dictionary and add both to it.
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

tab_filepath = os.path.join(report_dir, "theta2theta/Tables") # Defines a table filepath within the report directory

#######   Function Initialisation  ######

def extract_data_from_file(file_path):
    def is_comment(line):
        return line.startswith('#')

    with open(file_path, 'r') as f:
        skip_rows = 1

    data = pd.read_csv(file_path, sep=',', header=None, names=['theta', 'intensity'])
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    if data.empty:
        print(f"Warning: No valid numeric data found in {file_path}")
        return np.array([]), np.array([])

    theta = data['theta'].values
    original_intensity_corrected = data['intensity'].values

    return theta, original_intensity_corrected

################# Plot 1 ###########################


## Overlay Plot of theta2theta data, with two subplots, to simplify python have cut out using brokenaxes for this and am instead simply plotting within the range 35-65.

# Define subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Set colormap
cmap = mcolors.ListedColormap(["blue", "fuchsia", "indigo", "seagreen", "mediumblue", "indigo", "dodgerblue", "lime", "orange", "red"])
colours = [cmap(i) for i in range(cmap.N)]  # Extract colours from colormap
colour_cycle = iter(colours)

# Separate files into two categories
EH01_files = []
EH02_files = []

for filename in os.listdir(tab_filepath):
    if filename.endswith("background_subtracted_data.csv"):
        if 'EH01' in filename:
            EH01_files.append(filename)
        elif 'EH02' in filename:
            EH02_files.append(filename)




def plot_group(ax, file_list, title):
    colour_cycle = iter(colours)
    for filename in file_list:
        file_path = os.path.join(tab_filepath, filename)
        Colour = next(colour_cycle)

        sample_name_label = re.search(r"Sample_(.*)_background", filename)
        label_from_file = sample_name_label.group(1) if sample_name_label else filename

        theta, original_intensity_corrected = extract_data_from_file(file_path)

        ax.plot(theta, original_intensity_corrected, label=label_from_file, color=Colour, lw=1)

        print(file_path)

    ax.set_ylabel('Intensity (Counts)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plot_group(axes[0], EH01_files, "EH01")
plot_group(axes[1], EH02_files, "EH02")

axes[1].set_xlabel('2θ (Degrees)')

for ax in axes:
    for label, theta in bulk_peaks.items():
        ax.axvline(x=theta, linestyle='-', color='black', linewidth=1.5, zorder=2)
        ymin, ymax = ax.get_ylim()
        label_position = ymax * 1.015
        ax.text(theta, label_position, label, color='black', rotation=90,
                verticalalignment='bottom', horizontalalignment='center', fontsize=7)

# Adjust layout
plt.tight_layout()
plt.xlim(35, 65)

# ##### Plot Graph: #####
# fig, axs = plt.subplots(figsize=(12, 6))  # Define fig and axs
# fig.subplots_adjust(hspace=0.4)  # Adjust spacing between subplots
#
# # Initialise brokenaxes with appropriate x-axis breaks
# brk = brokenaxes(xlims=((30, 32), (33, 65.5), (79.5, 85)), wspace=0.05)
#
# for filename in os.listdir(tab_filepath):
#     if filename.endswith("background_subtracted_data.csv"):
#         file_path = os.path.join(tab_filepath, filename)
#
#         Colour = next(colour_cycle)
#
#         # Select just the sample name from the filename
#         sample_name_label = re.search(r"Sample_(.*)_background", filename)
#         label_from_file = sample_name_label.group(1) # Group 1 is the (.*) section
#
#         # Call function which returns theta and original_intensity_corrected
#         theta, original_intensity_corrected = extract_data_from_file(file_path)
#
#         ##### Identify and Remove Si Peaks ####
#
#         # Identify the silicon peak at 69 degrees
#         silicon_peak_index_1 = np.argmax(original_intensity_corrected)
#         silicon_peak_theta_1 = theta[silicon_peak_index_1]
#         silicon_peak_intensity = original_intensity_corrected[silicon_peak_index_1]
#
#         # For the second peak, define the theta range for the search
#         theta_min, theta_max = 30, 35
#         in_range = (theta >= theta_min) & (theta <= theta_max)
#
#         # Identify the silicon peak at 33 degrees
#         silicon_peak_index_2 = np.argmax(original_intensity_corrected[in_range])
#         silicon_peak_theta_2 = theta[silicon_peak_index_2]
#         silicon_peak_intensity = original_intensity_corrected[silicon_peak_index_2]
#
#         # Set the location and ranges of the peaks for masking and for the graph break.
#         peak_1_min = 65.5
#         peak_1_max = 79.5
#
#         peak_2_range = 0.2
#         peak_2_min = silicon_peak_theta_2 - peak_2_range
#         peak_2_max = silicon_peak_theta_2 + peak_2_range
#
#         # Remove the silicon peak by setting values to NaN in a small range around the peak.
#         mask_1 = (theta >= peak_1_min) & (theta <= peak_1_max)
#         original_intensity_corrected[mask_1] = np.nan
#
#         mask_2 = (theta >= peak_2_min) & (theta <= peak_2_max)
#         original_intensity_corrected[mask_2] = np.nan
#
#         # Plot the background-subtracted original unbinned intensity after background subtraction
#         brk.plot(theta, original_intensity_corrected, label=label_from_file, color=Colour, lw=1)
#
# # Plot the bulk peak lines
# for label, theta in bulk_peaks.items():
#     # Plot the bulk line
#     brk.axvline(x=theta, linestyle='-', color='black', linewidth=1.5, zorder=2)
#
#     # Determine which subplot the peak belongs to
#     for ax in brk.axs:
#         xlim = ax.get_xlim()
#         if xlim[0] <= theta <= xlim[1]:
#             # Get the current y-axis limits for the subplot
#             ymin, ymax = ax.get_ylim()
#             label_position = ymax * 1.05  # Adjust label position as needed
#
#             # Add the label within the corresponding subplot
#             ax.text(
#                 theta, label_position, label, color='black', rotation=90,
#                 verticalalignment='bottom', horizontalalignment='center', fontsize=7
#             )
#             break  # Label placed, no need to check other subplots
#
# # Plot lines where the original theta peaks were.
# brk.axvline(32.95, color='r', linestyle='', label=f'Silicon Peak at 32.95°')
# brk.axvline(69.13, color='r', linestyle='', label=f'Silicon Peak at 69.13°')
#
# # Set parameters for Graph
# brk.set_xlabel('2θ (Degrees)')
# brk.set_ylabel('Intensity (Counts)')
# brk.legend()
# brk.grid(True)

# Save the plot of background-subtracted original unbinned data
fin_filename = 'theta2theta_overlay.png'
fin_filepath = os.path.join(report_dir, "theta2theta/Graphs", fin_filename)
plt.savefig(fin_filepath)
print("Plot of overlay theta2theta data saved as 'theta2theta_overlay.png'")

