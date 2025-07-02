import deeplabcut

config_path = r"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\config.yaml"

deeplabcut.extract_frames(
    config_path, mode="automatic", algo="kmeans",
    crop=False,userfeedback=False
)

