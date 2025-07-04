import deeplabcut
import os
import glob

config_path = r"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\config.yaml"
# videodir_path = r"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\labeled-data_forAnalyze"
videodir_path = r"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\labeled-data"

video_files = glob.glob(os.path.join(videodir_path, "*.avi"))

deeplabcut.extract_outlier_frames(config_path, video_files,
                                  outlieralgorithm="jump",
                                  shuffle=3, epsilon=10)