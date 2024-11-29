import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv
import os
import sys

#########    File Path:    ############
# Change to needed file
#file_path = 'EOH1_2theta.txt'

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

##### Access Bash Script File Path: #####
directory = sys.argv[1]  # The first argument - (Data folder)
print(directory)
graphs_dir = sys.argv[2] # Second - graphs folder
table_dir = sys.argv[3] # Third - Values folder
sample_name = sys.argv[4] # Fourth - sample name

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)

###### Expected Peak Position #######
# Look at graph and find approximate peak position
expected_peaks = [38.0, 43.0, 55, 61.78, 64.5, 77.74]
range_width = 2.0  # +/- range to remove around each expected peak, can vary if needed but working like this

##########  Bin Size #########

bin_size = 5  # Adjust bin size as needed, remember the higher the bin size the more data loss

#### Key ####

# Plot 1: Masked Data
# Plot 2: Combined fitting plot
# Plot 3: Presentation data with guassian
# Table 1: Saved Guassian fit data

#######   Function Initialisation  ######

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

def rebin(data, n):
    max_bin_size = len(data) - (len(data) % n)
    if max_bin_size == 0:
        return data
    return np.mean(data[:max_bin_size].reshape(-1, n), axis=1)

def mask_peaks(theta, intensity, expected_peaks, range_width):
    mask = np.ones(intensity.shape, dtype=bool)

    for peak in expected_peaks:
        mask[np.abs(theta - peak) <= range_width] = False

    masked_theta = theta[mask]
    masked_intensity = intensity[mask]

    return masked_theta, masked_intensity

# Define an exponential background function
def exp_background(theta, a, b, c):
    return a * np.exp(-b * theta) + c

# Define Gaussian function for peak fitting
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

# Fit Gaussian to the data around a given peak
def fit_gaussian(theta, intensity, peak_pos, width=2.0):
    mask = (theta >= peak_pos - width) & (theta <= peak_pos + width)
    theta_fit = theta[mask]
    intensity_fit = intensity[mask]

    initial_guess = [max(intensity_fit), peak_pos, 1.0]  # [amplitude, mean, sigma]

    try:
        popt, pcov = curve_fit(gaussian, theta_fit, intensity_fit, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))  # Errors on the fit parameters
        return popt, perr
    except RuntimeError:
        return None, None

####  Data Extraction and Rebinning #####

# Extract data from the file
theta, intensity = extract_data_from_file(file_path)

# Rebin the data
rebinned_theta = rebin(theta, bin_size)
rebinned_intensity = rebin(intensity, bin_size)

#### Peak Masking in Background fit ####

# Masks the peaks so that the exponential background can be properly fitted without peaks interfering in fit, the expected peaks were defined up there and are done by eye for each graph.

# Mask the peaks in the rebinned dataset
masked_theta, masked_intensity = mask_peaks(rebinned_theta, rebinned_intensity, expected_peaks, range_width)

# Check if there's enough data left after masking for a reliable fit
if len(masked_theta) < 3:
    raise ValueError("Not enough data points left after masking to perform a fit.")

####  Exponential Background Fitting ####

# Fit the background using the masked data
initial_guess = [1, 0.1, 1]  # Adjust initial parameters as needed for exponential fit
params, _ = curve_fit(exp_background, masked_theta, masked_intensity, p0=initial_guess)

# Generate the fitted background for the full range using the original data
fitted_background = exp_background(theta, *params)

# Subtract the fitted background from the rebinned intensity
rebinned_fitted_background = exp_background(rebinned_theta, *params)
rebinned_intensity_corrected = rebinned_intensity - rebinned_fitted_background

##### Initial Peak detection ######

# Rough estimate of peaks found for use later in the guassian fitting, so that knows where to fit guassian, uses the rebinned data because otherwise you end up with millions of peaks

# Peak detection using rebinned data
peaks, properties = find_peaks(rebinned_intensity_corrected, height=10, distance=5, prominence=5, width=3)

# Get the peak positions and peak intensities for rebinned data
rebinned_peak_positions = rebinned_theta[peaks]  # 2θ values where peaks occur
rebinned_peak_heights = properties["peak_heights"]  # Intensities of the peaks

###  Guassian Fit for detected peaks ###

# More specific peak detection by fitting a guassian to refine peak placement and get errors

# Fit Gaussian around each detected peak and calculate errors
gaussian_fits = []
for i, peak_pos in enumerate(rebinned_peak_positions):
    popt, perr = fit_gaussian(rebinned_theta, rebinned_intensity_corrected, peak_pos)
    if popt is not None:
        gaussian_fits.append((popt, perr))

############  Plotting ###############

#### Plot 1 ###

# Masked Data and fitted background:
# Used for lab book and to show what the background fit is actually fitting to. Visual indication of how much data has been removed by data masking.

plt.figure(figsize=(10, 5))

# Plot masked data used for fitting
plt.scatter(masked_theta, masked_intensity, color='orange', label='Masked Data for Fitting')

# Fitted exponential background for the masked data
plt.plot(masked_theta, exp_background(masked_theta, *params), label=f'Fitted Background: {params[0]:.2f}*exp(-{params[1]:.2f}*θ) + {params[2]:.2f}', color='r', linestyle='--')

# Labels and title
plt.xlabel('2θ (Degrees)')
plt.ylabel('Intensity (Counts)')
plt.title('Masked Data for Background Fitting')
plt.legend()
plt.grid(True)

# Save the masked data plot
filename_masked = 'masked_data_and_fitted_background.png'
output_path_masked = os.path.join(graphs_dir, filename_masked)
plt.savefig(output_path_masked)
print("Masked data and fitted background plot saved as 'masked_data_and_fitted_background.png'")

######  Plot 2:  #######

# Original Data with fitted exponential decay and background subtracted rebinned data with peak positions indicated

plt.figure(figsize=(12, 6))

# Original intensity data (before background subtraction)
plt.plot(theta, intensity, label='Original Data', color='b', lw=1)

# Re-binned corrected intensity after background subtraction
plt.plot(rebinned_theta, rebinned_intensity_corrected, label='Rebinned Background Subtracted Data', color='magenta', lw=2)

# Fitted exponential background for rebinned data
plt.plot(rebinned_theta, rebinned_fitted_background,
         label=f'Fitted Background: {params[0]:.2f}*exp(-{params[1]:.2f}*θ) + {params[2]:.2f}',
         color='r', linestyle='--')

# Plot individual Gaussian fits in the same color
first_gaussian = True
for popt, perr in gaussian_fits:
    if first_gaussian:
        plt.plot(rebinned_theta, gaussian(rebinned_theta, *popt), color='black', label='Fitted Gaussian')
        first_gaussian = False
    else:
        plt.plot(rebinned_theta, gaussian(rebinned_theta, *popt), color='black')

# Highlight detected peaks using Gaussian-defined positions
detected_peak_positions = [popt[1] for popt, _ in gaussian_fits]  # Get mean (peak position) from Gaussian fits
detected_peak_heights = [gaussian(pos, *popt) for pos in detected_peak_positions for popt, _ in gaussian_fits if pos == popt[1]]

# Place red 'x' markers at the Gaussian-defined detected peak positions
plt.scatter(detected_peak_positions, detected_peak_heights, color='red', marker='x', s=100, label='Detected Peaks', zorder=5)

# Add vertical lines at detected peak positions
for pos in detected_peak_positions:
    plt.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.7)  # Added alpha for visibility

# Labels and title
plt.xlabel('2θ (Degrees)')
plt.ylabel('Intensity (Counts)')
plt.title('Original and Rebinned XRD Data with Background Subtraction and Detected Peaks')
plt.legend()
plt.grid(True)

# Save the plot with original and rebinned data
filename_combined = 'combined_plot_original_plus_background_corrected.png'
output_path_combined = os.path.join(graphs_dir, filename_combined)
plt.savefig(output_path_combined)
print("Combined plot of original and rebinned data saved as 'combined_plot_original_plus_background_corrected.png'")


##########  Table 1:  ############

# A table of the guassians that are shown in the plots, with the position of the guassian, amplitude of the guassian, the mean 2theta angle, sigma and errors.

# Define the CSV file name
csv_filename = "fit_err_guassian_table.csv"
csv_filepath = os.path.join(table_dir, csv_filename)

# Open the CSV file for writing
with open(csv_filepath, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Peak', 'Amplitude', 'Mean (2θ)', 'Sigma', 'Error (Mean)', 'Error (Sigma)'])

    # Write the data rows
    for i, (popt, perr) in enumerate(gaussian_fits):
        writer.writerow([f'fitted gaussian {i+1}', f'{popt[0]:.4f}', f'{popt[1]:.4f}', f'{popt[2]:.4f}', f'{perr[1]:.4f}', f'{perr[2]:.4f}'])

print(f"Table of Gaussian parameters saved as {csv_filename}")

######### Saving Processed Data #########

# Saves the background removed data to a .csv file with the same name as the original file name. Has two columns, theta and intensity

# --- Save Background Subtracted Original Data ---
background_subtracted_filename = "background_subtracted_data.csv"
original_intensity_corrected = intensity - fitted_background
background_subtracted_filepath = os.path.join(table_dir, background_subtracted_filename)
pd.DataFrame({'theta': theta, 'intensity': original_intensity_corrected}).to_csv(background_subtracted_filename, index=False)
print(f"Background subtracted data saved as {background_subtracted_filename}")

##########  Plot 3: ############

## Plot of the Background subtracted but unbinned data, with an overlay of the guassians to fit the peaks.

# --- Plot 3: Background-Subtracted Original Unbinned Data and Gaussians ---
plt.figure(figsize=(12, 6))

# Corrected original unbinned intensity after background subtraction
plt.plot(theta, original_intensity_corrected, label='Background Subtracted Original Data', color='b', lw=1)

# Plot individual Gaussian fits over the unbinned data
for popt, perr in gaussian_fits:
    plt.plot(theta, gaussian(theta, *popt), color='black')

# Highlight detected peaks using the mean from Gaussian fits
detected_peak_positions = [popt[1] for popt, _ in gaussian_fits]  # Get mean (peak position) from Gaussian fits

# Calculate the heights of the detected peaks based on their fitted Gaussian parameters
detected_peak_heights = [gaussian(pos, *popt) for pos in detected_peak_positions for popt, _ in gaussian_fits if pos == popt[1]]

# Check if the lengths of detected_peak_positions and detected_peak_heights match
if len(detected_peak_positions) != len(detected_peak_heights):
    print("Warning: The number of detected peak positions and heights do not match!")

# Place red 'x' markers at the detected peak positions
plt.scatter(detected_peak_positions, detected_peak_heights, color='red', marker='x', s=100, zorder=5)

# Add a legend entry for each detected peak position
for pos, height in zip(detected_peak_positions, detected_peak_heights):
    plt.plot(pos, height, 'rx', markersize=10, label=f'Peak at {pos:.2f}°', linestyle='None')  # Invisible but labeled

# Add vertical lines at detected peak positions
for pos in detected_peak_positions:
    plt.axvline(x=pos, color='red', linestyle='--', linewidth=1.75, alpha=0.7, zorder=3)  # Vertical lines at peak positions

# Labels and title
plt.xlabel('2θ (Degrees)')
plt.ylabel('Intensity (Counts)')
#plt.title('Background Subtracted Original Unbinned Data with Fitted Gaussians')
for label, theta in bulk_peaks.items():
    #plt.plot([theta, theta], [-40, 100], linestyle='-', color='magenta', label=label)
    # Plot the bulk line
    plt.axvline(x=theta, linestyle='-', color='darkmagenta', linewidth=1.5, zorder=2)

    # Get the current y-axis limits
    ymin, ymax = plt.ylim()

    # Place the label based on y-axis limits
    #  Place the label near the top of the plot, can be adjusted up and down.
    label_position = ymax * 1.015  # Adjust if needed; 1 = top, 0 = bottom
    plt.text(theta, label_position, label, color='black', rotation=90,
             verticalalignment='bottom', horizontalalignment='center', fontsize = 7)
plt.legend()
plt.grid(True)

# Save the plot of background-subtracted original unbinned data with Gaussians
fin_filename = 'background_subtracted_original_unbinned_with_gaussians.png'
fin_filepath = os.path.join(graphs_dir, fin_filename)
plt.savefig(fin_filepath)
print("Plot of background-subtracted unbinned data with Gaussians saved as 'background_subtracted_original_unbinned_with_gaussians.png'")
