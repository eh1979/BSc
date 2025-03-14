import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import numpy as np
import warnings
import random
import sys

# Input the amount to zoom the graph by.  This is zoom_value*Hc so it's how much more than Hc you want it to be.
zoom_value = 1.2

warnings.filterwarnings("ignore")
plt.rcParams['axes.formatter.useoffset'] = False  # Gets rid of a weird offset when the data is plotted

# Get the directory from command-line argument (passed by bash script)
directory = sys.argv[1]  # The first argument passed to the script (Data folder)
sample_name = sys.argv[4]
graphs_dir = sys.argv[2]
table_dir = sys.argv[3]

# Initialise an empty data frame to store results for Hc and Hex
blocking = pd.DataFrame()
# Initialise an empty data frame to store the Hc_Left and Hc_right in.
HcLR = pd.DataFrame()

def calculate_hc_hex_from_file1(H, M, min_distance=150):
    # Find where magnetisation crosses zero
    zero_crossings = np.where(np.diff(np.sign(M)))[0]

    # Initialise variables for Hc and Hex
    Hc = None
    Hex = None

    if len(zero_crossings) >= 2:
        # Loop through the zero crossings to find pairs that are far enough apart
        for i in range(len(zero_crossings) - 1):
            Hc_left = H[zero_crossings[i]]
            Hc_right = H[zero_crossings[i + 1]]

            # Check if the distance between Hc_left and Hc_right is greater than the threshold
            if abs(Hc_right - Hc_left) >= min_distance:
                Hc = (Hc_right - Hc_left) / 2
                Hex = (Hc_left + Hc_right) / 2
                break  # Once a valid pair is found, break the loop
    return Hc, Hex, Hc_left, Hc_right


# Function to plot the hysteresis loop from file 2 with zoom on central region
def plot_hysteresis(file_path, plot_data, Hc, Hex):

    plt.figure(figsize=(8, 6))

    # Plot the data points with round dots and blue line
    plt.plot(plot_data['H'], plot_data['m'], 'bo-', markersize=5, label="Magnetisation")  # Blue with round dots
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Plot Hc and Hex if they are calculated
    if Hc is not None:
        plt.axvline(x=Hc, color='blue', linestyle='--', alpha=0, label=f'Hc = {Hc:.0f} Oe')
    if Hex is not None:
        plt.axvline(x=Hex, color='green', linestyle='--', label=f'Hex = {Hex:.0f} Oe')

    # Set the zoomed-in limits for the x-axis (field)
    plt.xlim(zoom_min, zoom_max)

    # Add a grid to the plot
    plt.grid(True)

    plt.xlabel("H (Oe)")
    plt.ylabel("M/M$_{S}$")
    plt.title(f'Hysteresis Loop - {os.path.basename(file_path)}')
    plt.legend()

   # Save the plot with the name of the file
    output_filename = f'{os.path.splitext(os.path.basename(file_path))[0]}_hysteresis_zoomed.png'
    output_path = os.path.join(graphs_dir, output_filename)
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid displaying each one
    print(f"Zoomed plot saved as '{output_filename}'")

# Initialise a list to hold the file data for the overlay graph
overlay_data = []

# Loop through all files in the current directory
for filename in os.listdir(directory):
    if filename.endswith(".VHD"):
        file_path = os.path.join(directory, filename)
        title = file_path

        data = pd.DataFrame()  # Create a dataframe for the file from the VSM to go into

        # Load the .VHD file
        with open(file_path, 'r') as f:
            for line in f:
                data = pd.concat([data, pd.DataFrame([tuple(line.strip().split('   '))])], ignore_index=True)

        # Find the start and end points of the data
        x = data[data.isin(['Applied_Field_For_Plot_']).any(axis=1)]
        y = x.index[0] + 4
        x2 = data[data.isin(['@@END Data.']).any(axis=1)]
        y2 = x2.index[0] - 1

        totdat = y2 - y
        ddat1 = int(totdat / 2 - 7)
        ddat2 = int(totdat / 2 + 7)

        plot_data = pd.DataFrame()  # Create a dataframe for the final data we actually need
        plot_data['H'] = data.iloc[y:y2, 8].astype(float)  # Applied Field
        plot_data['m_x'] = data.iloc[y:y2, 11].astype(float)  # Magnetisation x-component
        plot_data['m_y'] = data.iloc[y:y2, 12].astype(float)  # Magnetisation y-component

        # Data analysis for m_x
        slope, intercept, rvalue, pvalue, stderr = linregress(plot_data.iloc[ddat1:ddat2, 0], plot_data.iloc[ddat1:ddat2, 1])
        plot_data['m_corr'] = plot_data['m_x'] - (slope * plot_data['H'])
        shift = (plot_data.iloc[ddat1:ddat2, 3].mean() + plot_data.iloc[1:5, 3].mean()) / 2
        plot_data['m_corr_offset'] = plot_data['m_corr'] - shift
        plot_data['M/Ms'] = plot_data['m_corr_offset'] / plot_data.iloc[1:10, 4].mean()

        # Data analysis for m_y
        slope, intercept, rvalue, pvalue, stderr = linregress(plot_data.iloc[ddat1:ddat2, 0], plot_data.iloc[ddat1:ddat2, 2])
        plot_data['m_corr_y'] = plot_data['m_y'] - (slope * plot_data['H'])
        shift = (plot_data.iloc[ddat1:ddat2, 2].mean() + plot_data.iloc[1:10, 2].mean()) / 2
        plot_data['m_corr_offset_y'] = plot_data['m_corr_y'] - shift
        plot_data['My/Ms'] = plot_data['m_corr_y'] / plot_data['m_corr_offset'].max()

        plot_data.to_csv(file_path + "_plotdata.csv")  # Save it as a .csv file

        # Data processing for Hc and Hex
        data = pd.read_csv(file_path + "_plotdata.csv", skiprows=0, usecols=[1, 6], skip_blank_lines=True, delimiter=',', encoding='unicode_escape')

        plot_data = pd.DataFrame()
        plot_data['H'] = data.iloc[:, 0]
        plot_data['m'] = data.iloc[:, 1]  # This is the magnetisation data

        Hc, Hex, Hc_left, Hc_right = calculate_hc_hex_from_file1(plot_data['H'].values, plot_data['m'].values)

        # Round Hc and Hex to integers
        if Hc is not None:
            Hc = round(Hc)
        if Hex is not None:
            Hex = round(Hex)

         # Store data for overlay graph
        overlay_data.append({
            'H': plot_data['H'],
            'm': plot_data['m'],
            'filename': filename
        })

        calc = {}  # Initialise calc as a temporary dictionary to store the results for each file

        calc['Hc_left'] = Hc_left
        calc['Hc_right'] = Hc_right

        # Append the results for this file to the Hc Left and Hc Right data frame
        HcLR = pd.concat([HcLR, pd.DataFrame([calc])], ignore_index=True)

        zoom_min = Hc_left - zoom_value*Hc
        zoom_max = Hc_right + zoom_value*Hc


        # Plot the hysteresis loop with zoom on central data from file 2
        plot_hysteresis(file_path, plot_data, Hc, Hex)


        # Store results for Hc and Hex as integers
        temp = {}  # Initialise temp as a dictionary to store the results for each file
        # Extract just the last section of the filename, remove the '.VHD' extension
        file_name_last_section = filename.split('-')[-1].replace('.VHD', '')
        temp['Filename'] = file_name_last_section  # Add the processed filename here
        temp['Hc'] = Hc
        temp['Hex'] = Hex

        # Append the results for this file to the blocking data frame
        blocking = pd.concat([blocking, pd.DataFrame([temp])], ignore_index=True)

# Overlay min and max X limits -> largest and smallest Hc values so that all graphs are displayed.

overlay_Hc_left = HcLR['Hc_left'].min()
overlay_Hc_right = HcLR['Hc_right'].max()

overlay_zoom_min = overlay_Hc_left - zoom_value*Hc
overlay_zoom_max = overlay_Hc_right + zoom_value*Hc

# Overlay graph - Plot all the hysteresis loops on top of each other

plt.figure(figsize=(8, 6))
color_list = ['blue', 'purple', 'magenta']  # Define colors for each dataset

for i, data in enumerate(overlay_data):

    # Determine marker based on the file name
    if "Initial" in data['filename']:
        marker = 'o-'  # Circular for samples before annealing
        if "Annealing" in data['filename']:
            name = 'Initial Annealed 0C'
        else:
            name = f'Initial Annealed {data['filename'].split('_')[-1].replace(".VHD", "")}'
        #Changes the name to reflect annealing temperature.
    elif "100C" in data['filename']:
        marker = '^-'  # Triangular for thermally activated samples
        if "Annealing" in data['filename']:
            name = '100C Annealed 0C'
        else:
            name = f'100C Annealed {data['filename'].split('_')[-1].replace(".VHD", "")}'
    else:
        marker = 's-'  # Square for who knows what errors
        
    # Use a specified color or random color for each dataset's plot
    color = color_list[i] if i < len(color_list) else (random.random(), random.random(), random.random())

     # Shorten filename by removing the ".VHD" extension and using only the last part after '-'
    label_name = name
    #label_name = data['filename'].split('-')[-1].replace(".VHD", "")

    # # Plot data with marker assigned in section above, using the colour from the color thing earlier.
    plt.plot(data['H'], data['m'], marker, markersize=5, label=label_name, color=color)

    # Retrieve Hc and Hex for each dataset
    Hc = blocking.loc[i, 'Hc']
    Hex = blocking.loc[i, 'Hex']

    # Plot Hc and Hex lines with the same color as the dataset's plot
    if pd.notna(Hc):
        plt.axvline(x=Hc, color=color, linestyle='--',alpha=0)
#, label=f'{label_name} Hc = {Hc} Oe'

    if pd.notna(Hex):
        plt.axvline(x=Hex, color=color, linestyle='--')
#label=f'{label_name} Hex = {Hex} Oe'


# Set axis labels and title
plt.xlabel("H (Oe)")
plt.ylabel("M/M$_{S}$")
plt.title("Overlay of Hysteresis Loops")

# Changes to the size of the hysteresis loops by zooming in around centre of loops. Size multiplied to allow room for legend.
plt.xlim(overlay_zoom_min*1.2, overlay_zoom_max*1.2)

# Bold the central axes for the overlay graph with lighter emphasis
plt.axhline(0, color='black', linewidth=1.5)  # Slightly bolder central x-axis
plt.axvline(0, color='black', linewidth=1.5)  # Slightly bolder central y-axis

# Add a grid to the overlay plot
plt.grid(True)

# Show the legend with smaller font size to reduce overlap, location in the upper left, if needed for file can change to loc='best' or for slightly larger text loc='small'
plt.legend(fontsize=8, loc='upper left', ncol=1)

# Save the overlay plot
overlay_filename = "overlay_hysteresis_loops.png"
output_path_overlay = os.path.join(graphs_dir, overlay_filename)
plt.savefig(output_path_overlay)
plt.close()
print(f"Overlay plot saved as '{overlay_filename}'")


# Save the final results

output_table_path = os.path.join(table_dir, 'Hc_Hex_results.csv')
# Save the final table with all results to a CSV file
blocking.to_csv(output_table_path, index=False)
print("Blocking results saved to 'blocking_results.csv'")
