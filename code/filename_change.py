import os
import glob

def rename_files(directory, old_prefix, new_prefix):
    for filename in os.listdir(directory):
        new_filename = filename.replace(old_prefix, new_prefix)
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

old_prefix = "_Cladonema_starved_crop_"
new_prefix = "_"
directory_path = "D:\Satiety_differentially_modulates_feeding_\dlc\Cladonema_starved_tentacle20-Izuki-2025-06-26\DSC_3204_#1\dlc_label"
rename_files(directory_path, old_prefix, new_prefix)

