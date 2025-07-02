import h5py
import pandas as pd

h5_path= r"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\original_labeled-data\DSC_3175_roi_trial2_crop\CollectedData_Izuki.h5"
with h5py.File(h5_path, 'r') as f:
    print(list(f.keys()))

df = pd.read_hdf(h5_path)

print(df.head())