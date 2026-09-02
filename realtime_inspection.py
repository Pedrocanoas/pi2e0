import os
import time
import threading

import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = "best26s_ncnn_model"

CAMERA_URL = "http://127.0.0.1:8080/video"

ROI = (112, 47, 1146, 720)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMGSZ = 320
CONF_PREDICT = 0.4

DISPLAY_MAX_WIDTH = 1200

TRACKER_CFG = "bytetrack_custom.yaml"
CONFIRMATION_THRESHOLD = 4
TRACK_TIMEOUT_FRAMES = 130
MAX_MERGE_DIST_PX = 150

RESULT_DIR = os.path.join("runs", "results")

DEBUG_SAVE_FRAMES = False
DEBUG_DIR = os.path.join("runs", "debug_frames")

running = True
latest_frame = None
annotated_frame = None
frame_lock = threading.Lock()

frame_count = 0
track_stats = {}
confirmed_defects = []
byte_to_canonical = {}


def resolve_canonical_id(byte_id, center, current_frame):
    if byte_id in byte_to_canonical:
        return byte_to_canonical[byte_id]

    best_id, best_dist = None, MAX_MERGE_DIST_PX
    for cid, stat in track_stats.items():
        if stat["center"] is None:
            continue
        if current_frame - stat["last_seen"] > TRACK_TIMEOUT_FRAMES:
            continue
        dist = ((stat["center"][0] - center[0]) ** 2 + (stat["center"][1] - center[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_id, best_dist = cid, dist

    canonical_id = best_id if best_id is not None else byte_id
    byte_to_canonical[byte_id] = canonical_id
    return canonical_id


def crop_roi(frame, roi=ROI):
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def resize_for_display(frame, max_width=DISPLAY_MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def capture_thread(cap):
    global latest_frame, running

    while running:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame = frame


def inference_thread(model):
    global annotated_frame, running, frame_count

    debug_log_path = os.path.join(DEBUG_DIR, "log.csv")
    prev_frame_bytes = None
    prev_ts = None
    if DEBUG_SAVE_FRAMES:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(debug_log_path, "w") as f:
            f.write("frame_count,ts,gap_ms,same_bytes_as_prev,num_det,max_conf,raw_track_ids,recovered_track_ids\n")

    while running:
        with frame_lock:
            frame = crop_roi(latest_frame).copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        now = time.time()

        results = model.track(
            source=frame,
            conf=CONF_PREDICT,
            imgsz=IMGSZ,
            device=DEVICE,
            persist=True,
            tracker=TRACKER_CFG,
            verbose=False,
        )[0]

        tracker_obj = model.predictor.trackers[0]
        current_tracks = [t for t in tracker_obj.tracked_stracks if t.frame_id == tracker_obj.frame_id]

        if DEBUG_SAVE_FRAMES:
            frame_bytes = frame.tobytes()
            same_bytes = frame_bytes == prev_frame_bytes
            gap_ms = (now - prev_ts) * 1000 if prev_ts is not None else 0
            num_det = len(results.boxes) if results.boxes is not None else 0
            max_conf = float(results.boxes.conf.max()) if num_det else 0.0
            if num_det and results.boxes.id is not None:
                raw_track_ids = ";".join(str(int(t)) for t in results.boxes.id)
            else:
                raw_track_ids = ""
            recovered_track_ids = ";".join(str(t.track_id) for t in current_tracks)
            with open(debug_log_path, "a") as f:
                f.write(
                    f"{frame_count},{now:.3f},{gap_ms:.1f},{same_bytes},{num_det},{max_conf:.3f},"
                    f"{raw_track_ids},{recovered_track_ids}\n"
                )
            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"{frame_count:05d}_{'det' if num_det else 'nodet'}.jpg"),
                frame,
            )
            prev_frame_bytes = frame_bytes
            prev_ts = now

        annotated = frame.copy()

        if current_tracks:
            for t in current_tracks:
                x1, y1, x2, y2 = map(int, t.xyxy)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                track_id = resolve_canonical_id(t.track_id, center, frame_count)

                stat = track_stats.setdefault(
                    track_id,
                    {"hits": 0, "first_seen": frame_count, "confirmed": False, "center": None},
                )
                stat["hits"] += 1
                stat["last_seen"] = frame_count
                stat["center"] = center

                if stat["hits"] >= CONFIRMATION_THRESHOLD and not stat["confirmed"]:
                    stat["confirmed"] = True
                    confirmed_defects.append({
                        "track_id": track_id,
                        "hits": stat["hits"],
                        "frame": frame_count,
                    })
                    print(f"✅ Defect CONFIRMED (id={track_id}) after {stat['hits']} detections")
                    os.makedirs(RESULT_DIR, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(RESULT_DIR, f"confirmed_defect_{track_id}_{frame_count}.jpg"),
                        frame,
                    )

                confirmed = stat["confirmed"]
                color = (0, 0, 255) if confirmed else (0, 165, 255)
                label = f"DEFECT #{track_id} ({stat['hits']}/{CONFIRMATION_THRESHOLD})"
                if confirmed:
                    label += " CONFIRMED"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

        stale_ids = [
            tid for tid, s in track_stats.items()
            if frame_count - s["last_seen"] > TRACK_TIMEOUT_FRAMES
        ]
        for tid in stale_ids:
            s = track_stats[tid]
            tag = "CONFIRMED" if s["confirmed"] else "never confirmed"
            print(f"🔍 track {tid}: {s['hits']} hit(s) before leaving frame ({tag})")
            del track_stats[tid]
            for byte_id in [k for k, v in byte_to_canonical.items() if v == tid]:
                del byte_to_canonical[byte_id]

        with frame_lock:
            annotated_frame = annotated


def main():
    global running

    print(f"📦 Loading trained model: {MODEL_PATH} (device={DEVICE}, imgsz={IMGSZ})")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to access the IP camera.")

    threads = [
        threading.Thread(target=capture_thread, args=(cap,), daemon=True),
        threading.Thread(target=inference_thread, args=(model,), daemon=True),
    ]
    for t in threads:
        t.start()

    print(f"✅ Camera connected. A defect is confirmed after {CONFIRMATION_THRESHOLD} detections.")
    print("   Press 'q' in the video window (or Ctrl+C) to stop.")

    cv2.namedWindow("Glass Defect Inspection", cv2.WINDOW_NORMAL)

    try:
        while True:
            with frame_lock:
                display_frame = annotated_frame if annotated_frame is not None else latest_frame

            if display_frame is not None:
                cv2.imshow("Glass Defect Inspection", resize_for_display(display_frame))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("🛑 Manually interrupted.")
    finally:
        running = False
        for t in threads:
            t.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()

        print(f"\n📋 {len(confirmed_defects)} defect(s) confirmed (>= {CONFIRMATION_THRESHOLD} detections each):")
        for d in confirmed_defects:
            print(f"   - id={d['track_id']}: detected {d['hits']}x (confirmed at frame {d['frame']})")
        print("👋 Finished.")


if __name__ == "__main__":
    main()
