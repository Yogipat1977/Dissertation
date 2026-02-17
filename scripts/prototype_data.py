import os
import random
import shutil
from pathlib import Path

# Set the source path (your current training data location)
# Note: Ensure there are no leading spaces in folder names
source_path = Path("/home/yogipatel/Desktop/Dissertation/data/BraTS2023_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")

# Set the destination path for your prototype
dest_path = Path("~/Desktop/Dissertation/data/Prototype_Data").expanduser()

# List all subdirectories (patient folders) in the training directory
all_patients = [f for f in source_path.iterdir() if f.is_dir()]

# Check if there are enough patients to select 50
if len(all_patients) >= 45:
    # Randomly select 40 unique patient folders
    selected_patients = random.sample(all_patients, 45)

    # Create the destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Copying 40 patients to {dest_path}...")

    for patient_folder in selected_patients:
        # Define the path for the new folder
        new_folder_path = dest_path / patient_folder.name
        # Copy the entire patient directory (all 5 .nii.gz files)
        shutil.copytree(patient_folder, new_folder_path, dirs_exist_ok=True)

    print("Selection and copying complete.")
else:
    print(f"Error: Only found {len(all_patients)} folders. Need at least 40.")
