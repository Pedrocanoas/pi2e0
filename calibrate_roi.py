import cv2

CAMERA_URL = "http://127.0.0.1:8080/video"

print(f"Connecting to {CAMERA_URL} ...", flush=True)
cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    raise SystemExit("FAILED to open stream.")

for _ in range(3):
    cap.read()

ret, frame = cap.read()
cap.release()
if not ret:
    raise SystemExit("FAILED to grab a frame.")

print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
print("Drag a rectangle over the area where the glass passes through, "
      "then press ENTER/SPACE to confirm (ESC to quit).")

x, y, w, h = cv2.selectROI("Select ROI - drag then ENTER", frame, showCrosshair=True)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise SystemExit("No ROI selected.")

x1, y1, x2, y2 = x, y, x + w, y + h
print(f"\nROI = ({x1}, {y1}, {x2}, {y2})")
print("Paste that line into realtime_inspection.py as the ROI constant.")

preview = frame[y1:y2, x1:x2]
cv2.imshow("Cropped preview - press any key to close", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
