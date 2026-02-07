from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
import time
import os
import json
import numpy as np
from datetime import datetime

app = Flask(__name__)

# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0

ROI_X1, ROI_Y1 = 200, 100
ROI_X2, ROI_Y2 = 500, 400

FPS = 10

# =========================
# INIT
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, FPS)

os.makedirs("recordings", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# =========================
# SHARED STATE
# =========================
raw_frame = None
annotated_frame = None
frame_lock = threading.Lock()

live_counts = {"person": 0}
intrusion_status = False

# Recording state
recording = False
video_writer = None
intrusion_start_time = None
event_counter = 0

fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# =========================
# HELPER: SAVE JSON LOG
# =========================
def save_log(start, end, duration, people):
    global event_counter

    event_counter += 1

    entry = {
        "event_id": f"evt_{event_counter:03d}",
        "start_time": start,
        "end_time": end,
        "duration_sec": round(duration, 2),
        "people_count": people
    }

    log_path = "logs/intrusion_logs.json"

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)

    print("[LOG SAVED]", entry)

# =========================
# CAMERA LOOP
# =========================
def camera_loop():
    global raw_frame, annotated_frame
    global intrusion_status, recording, video_writer, intrusion_start_time
    global live_counts

    prev_intrusion = False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        raw = frame.copy()
        results = model(frame, verbose=False)
        annotated = frame.copy()

        person_count = 0
        person_in_roi = False

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_count += 1
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if ROI_X1 < cx < ROI_X2 and ROI_Y1 < cy < ROI_Y2:
                    person_in_roi = True

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        live_counts["person"] = person_count
        intrusion_status = person_in_roi

        # Draw ROI
        cv2.rectangle(annotated, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

        if person_in_roi:
            cv2.putText(
                annotated,
                "⚠ INTRUSION DETECTED",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # =========================
        # RECORDING LOGIC
        # =========================
        if person_in_roi and not prev_intrusion:
            # Intrusion START
            intrusion_start_time = time.time()
            start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            filename = f"recordings/intrusion_{start_str.replace(':','-')}.mp4"
            video_writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                FPS,
                (frame.shape[1], frame.shape[0])
            )
            recording = True
            print("[RECORDING STARTED]")

        if recording:
            video_writer.write(raw)

        if not person_in_roi and prev_intrusion:
            # Intrusion END
            recording = False
            video_writer.release()
            video_writer = None

            end_time = time.time()
            end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration = end_time - intrusion_start_time

            save_log(start_str, end_str, duration, person_count)
            print("[RECORDING STOPPED]")

        prev_intrusion = person_in_roi

        with frame_lock:
            raw_frame = raw.copy()
            annotated_frame = annotated.copy()

        time.sleep(0.02)

# =========================
# STREAMS
# =========================
def stream(frame_type="annotated"):
    while True:
        with frame_lock:
            frame = annotated_frame if frame_type == "annotated" else raw_frame

        if frame is None:
            frame = fallback_frame

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

        time.sleep(0.05)

# =========================
# ROUTES
# =========================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/video_ai")
def video_ai():
    return Response(stream("annotated"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_raw")
def video_raw():
    return Response(stream("raw"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events")
def api_events():
    log_file = "logs/intrusion_logs.json"

    if not os.path.exists(log_file):
        return jsonify([])

    with open(log_file, "r") as f:
        data = json.load(f)

    return jsonify(data)
@app.route("/events")
def events():
    return render_template("events.html", title="Events")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", title="Analytics")

@app.route("/api/analytics")
def api_analytics():
    log_file = "logs/intrusion_logs.json"

    if not os.path.exists(log_file):
        return jsonify({
            "total_events": 0,
            "avg_duration": 0,
            "max_duration": 0,
            "total_duration": 0,
            "total_people": 0
        })

    with open(log_file, "r") as f:
        data = json.load(f)

    total_events = len(data)
    total_duration = sum(e["duration_sec"] for e in data)
    max_duration = max(e["duration_sec"] for e in data)
    avg_duration = total_duration / total_events if total_events > 0 else 0
    total_people = sum(e["people_count"] for e in data)

    return jsonify({
        "total_events": total_events,
        "avg_duration": round(avg_duration, 2),
        "max_duration": round(max_duration, 2),
        "total_duration": round(total_duration, 2),
        "total_people": total_people
    })

@app.route("/system")
def system():
    return render_template("system.html", title="System")

manual_warning = False

@app.route("/api/warn", methods=["POST"])
def api_warn():
    global manual_warning
    manual_warning = not manual_warning
    return jsonify({"manual_warning": manual_warning})

@app.route("/api/counts")
def api_counts():
    return jsonify({
        "person": live_counts["person"],
        "intrusion": intrusion_status or manual_warning
    })


# =========================
# START
# =========================
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(debug=False)
