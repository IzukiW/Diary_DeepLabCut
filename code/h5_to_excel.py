#準備
import numpy as np
import pandas as pd
import glob
import os
import openpyxl

#change directory
directory_path=r"Z:\evaluation-results-pytorch\iteration-0\Cladonema_starved_crop_tentacle20Jun26-trainset90shuffle3"
os.chdir(directory_path)

# Get all .h5 files in the directory
h5_files = glob.glob("*.h5")

for h5_file in h5_files:
    # Load the .h5 file
    data = pd.read_hdf(h5_file)

    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(h5_file))[0]

    # Save to excel
    excel_name = f"{filename}.xlsx"
    data.to_excel(excel_name)
    
    print(f"Converted {h5_file} to {excel_name}")