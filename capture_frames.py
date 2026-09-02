import os
import sys
import time

import cv2

from realtime_inspection import CAMERA_URL, ROI, crop_roi

OUT_DIR = os.path.join("runs", "captured_frames")
DURATION_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0

os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))

print(f"Connecting to {CAMERA_URL} ...", flush=True)
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("FAILED to open stream.")

for _ in range(3):
    cap.read()

print(f"Capturing for {DURATION_SEC:.0f}s into {OUT_DIR} - pass the glass through now...")
start = time.time()
i = 0
while time.time() - start < DURATION_SEC:
    ret, frame = cap.read()
    if not ret:
        continue
    i += 1
    roi_crop = crop_roi(frame, ROI)
    cv2.imwrite(os.path.join(OUT_DIR, f"{i:04d}_full.jpg"), frame)
    cv2.imwrite(os.path.join(OUT_DIR, f"{i:04d}_roi.jpg"), roi_crop)

cap.release()
print(f"✅ Saved {i} frames (full + roi each) to {OUT_DIR}")
