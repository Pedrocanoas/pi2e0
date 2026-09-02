import time
import numpy as np
from ultralytics import YOLO

from realtime_inspection import ROI, IMGSZ, CONF_PREDICT, TRACKER_CFG

roi_w = ROI[2] - ROI[0]
roi_h = ROI[3] - ROI[1]
dummy_frame = np.random.randint(0, 255, (roi_h, roi_w, 3), dtype=np.uint8)
print(f"Dummy frame size: {roi_w}x{roi_h} (matches current ROI)")

N = 20


def bench(name, model_path):
    print(f"\n--- {name} ({model_path}) ---")
    model = YOLO(model_path)

    model.track(source=dummy_frame, conf=CONF_PREDICT, imgsz=IMGSZ,
                device="cpu", persist=True, tracker=TRACKER_CFG, verbose=False)

    times = []
    for i in range(N):
        t0 = time.perf_counter()
        model.track(source=dummy_frame, conf=CONF_PREDICT, imgsz=IMGSZ,
                    device="cpu", persist=True, tracker=TRACKER_CFG, verbose=False)
        dt = time.perf_counter() - t0
        times.append(dt)

    avg = sum(times) / len(times)
    print(f"avg={avg*1000:.0f}ms  min={min(times)*1000:.0f}ms  max={max(times)*1000:.0f}ms  (~{1/avg:.1f} FPS)")
    return avg


t_pt = bench("PyTorch CPU", "best26s.pt")
t_ncnn = bench("NCNN", "best26s_ncnn_model")

print(f"\n=== Result: NCNN is {t_pt / t_ncnn:.2f}x the speed of plain PyTorch ===")
