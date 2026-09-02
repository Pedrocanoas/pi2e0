import time
import cv2

CAMERA_URL = "http://127.0.0.1:8080/video"

print(f"Connecting to {CAMERA_URL} ...", flush=True)
t0 = time.perf_counter()
cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print(f"FAILED to open stream after {time.perf_counter()-t0:.2f}s", flush=True)
    raise SystemExit(1)
print(f"Opened in {time.perf_counter()-t0:.2f}s", flush=True)

for _ in range(3):
    cap.read()

N = 20
times = []
for i in range(N):
    t0 = time.perf_counter()
    ret, frame = cap.read()
    dt = time.perf_counter() - t0
    if not ret:
        print(f"frame {i}: read FAILED after {dt*1000:.0f}ms", flush=True)
        continue
    times.append(dt)
    print(f"frame {i}: {dt*1000:.0f}ms  shape={frame.shape}", flush=True)

cap.release()

if times:
    avg = sum(times) / len(times)
    print(f"\navg cap.read() time: {avg*1000:.0f}ms  (~{1/avg:.1f} FPS ceiling from network+decode alone)")
    print(f"min={min(times)*1000:.0f}ms max={max(times)*1000:.0f}ms")
