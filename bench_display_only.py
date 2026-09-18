import cv2

from realtime_inspection import CAMERA_SOURCE, open_camera, resize_for_display

print(f"Connecting to camera {CAMERA_SOURCE} ...", flush=True)
cap = open_camera()
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("FAILED to open stream.")

print("Showing live feed, no model/inference running. Press 'q' to stop.")
cv2.namedWindow("Capture-only test", cv2.WINDOW_NORMAL)

drop_count = 0
frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            drop_count += 1
            print(f"frame {frame_count}: read FAILED (drop #{drop_count})", flush=True)
            continue
        cv2.imshow("Capture-only test", resize_for_display(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{frame_count} frames read, {drop_count} failed reads.")
