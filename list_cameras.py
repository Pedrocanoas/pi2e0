import cv2

MAX_INDEX = 6

found = []
for i in range(MAX_INDEX):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        continue
    ret, frame = cap.read()
    if ret:
        found.append((i, frame.shape[1], frame.shape[0]))
        print(f"index {i}: opened, frame size {frame.shape[1]}x{frame.shape[0]}")
        cv2.imshow(f"index {i} - press any key for next", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()

print("\nFound cameras:", found)
print("Set CAMERA_SOURCE in realtime_inspection.py to the index that showed the GoPro feed.")
