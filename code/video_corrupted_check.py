import cv2

video_path = r"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\videos\DSC_3180_roi_trial3_crop.avi"


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

# Get total number of frames reported by metadata
frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Metadata reports {frame_count_meta} frames.")

frame_index = 0
corrupted = False

while True:
    ret, frame = cap.read()
    if not ret:
        if frame_index < frame_count_meta:
            print(f"Unexpected end or read error at frame {frame_index} before metadata end.")
            corrupted = True
        break
    if frame is None:
        print(f"Frame {frame_index} is None despite ret=True — possibly corrupted.")
        corrupted = True
        break
    frame_index += 1

cap.release()

# Final summary
if corrupted:
    print("Video may be corrupted.")
else:
    print("Video read successfully without detected corruption.")