#!/bin/bash

# Define the base folder path
BASE_FOLDER="/home/e74237/BSc/Data"

# Give sample name
SAMPLE_NAME=$1
SAMPLE_FOLDER="$BASE_FOLDER/$SAMPLE_NAME"
VSM_DATA_DIR="$SAMPLE_FOLDER/VSM/Data"
VSM_GRAPHS_DIR="$SAMPLE_FOLDER/VSM/Graphs"
VSM_VALUES_DIR="$SAMPLE_FOLDER/VSM/Values"
XRD_2THETACHI_DIR="$SAMPLE_FOLDER/XRD/2thetachi/Data"
XRD_2THETA_DIR="$SAMPLE_FOLDER/XRD/2theta/Data"
XRD_XRR_DIR="$SAMPLE_FOLDER/XRD/XRR/Data"
XRD_THETA2THETA_DIR="$SAMPLE_FOLDER/XRD/theta2theta/Data"

fol_graphlder=("Graphs" "Values")

for fol in "${fol_graphlder[@]}"; do
    # Supposed to be Graph folder locations
    dir_graphlder=(
    "$SAMPLE_FOLDER/VSM/$fol"
    "$SAMPLE_FOLDER/XRD/2thetachi/$fol"
    "$SAMPLE_FOLDER/XRD/2theta/$fol"
    "$SAMPLE_FOLDER/XRD/XRR/$fol"
    "$SAMPLE_FOLDER/XRD/theta2theta/$fol"
    )

    # Loop through directories and create 'Graphs' or 'values' if they don't exist
    for dir in "${dir_graphlder[@]}"; do
    mkdir -p "$dir"  # Create the directory if it doesn't exist
    done
done

# Check if the sample folder exists
if [ ! -d "$SAMPLE_FOLDER" ]; then
  echo "Sample folder does not exist: $SAMPLE_FOLDER"
  exit 1
else
  echo "Using sample folder: $SAMPLE_FOLDER"
fi

# Python scripts to use for different types of data, eg. VSM, theta2theta, etc.
SCRIPTS=("VSM_full.py" "full_fitting_file_2theta.py" "full_fitting_file_2thetachi.py" "simple_plotter_XRR.py" "theta2theta_plotter.py")

# Run the VSM processing script
echo "Running VSM processing..."
python3 /home/e74237/BSc/Data/Template/${SCRIPTS[0]} "$VSM_DATA_DIR" "$VSM_GRAPHS_DIR" "$VSM_VALUES_DIR" "$SAMPLE_NAME"

# Run the XRD processing scripts for 2theta, 2thetachi, XRR, and theta2theta
echo "Running XRD processing..."

# Define a function to run the XRD scripts for different directories
run_xrd_processing() {
  local dir=$1
  local script=$2
  local graphs_dir=$3
  local values_dir=$4

  if [ -d "$dir" ]; then
    echo "Found XRD data in $dir"
    python3 /home/e74237/BSc/Data/Template/$script "$dir" "$graphs_dir" "$values_dir" "$SAMPLE_NAME"
  else
    echo "No XRD data found in $dir"
  fi
}

# Call the function for each XRD directory with the corresponding script and output directories
run_xrd_processing "$XRD_2THETACHI_DIR" "${SCRIPTS[2]}" "$SAMPLE_FOLDER/XRD/2thetachi/Graphs" "$SAMPLE_FOLDER/XRD/2thetachi/Values"
run_xrd_processing "$XRD_2THETA_DIR" "${SCRIPTS[1]}" "$SAMPLE_FOLDER/XRD/2theta/Graphs" "$SAMPLE_FOLDER/XRD/2theta/Values"
run_xrd_processing "$XRD_XRR_DIR" "${SCRIPTS[3]}" "$SAMPLE_FOLDER/XRD/XRR/Graphs" "$SAMPLE_FOLDER/XRD/XRR/Values"
run_xrd_processing "$XRD_THETA2THETA_DIR" "${SCRIPTS[4]}" "$SAMPLE_FOLDER/XRD/theta2theta/Graphs" "$SAMPLE_FOLDER/XRD/theta2theta/Values"

echo "Processing completed."
