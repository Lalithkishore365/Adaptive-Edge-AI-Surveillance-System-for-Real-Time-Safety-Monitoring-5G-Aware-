from flask import Flask, render_template, Response, jsonify, request
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
PORT = 8000

# Match Pi FPS (IMPORTANT)
FPS = 2

# Restricted Area (ROI)
ROI_X1, ROI_Y1 = 200, 100
ROI_X2, ROI_Y2 = 500, 400

os.makedirs("recordings", exist_ok=True)
os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/intrusion_logs.json"

# =========================
# INIT MODEL
# =========================
model = YOLO(MODEL_PATH)
print("✅ YOLO Loaded on Cloud Server")

# =========================
# SHARED STATE
# =========================
raw_frame = None              # RAW feed (Live page)
annotated_frame = None        # AI feed (Dashboard)
frame_lock = threading.Lock()

live_counts = {"person": 0}
intrusion_status = False

# Recording state
recording = False
video_writer = None
prev_intrusion = False

intrusion_start_time = None
intrusion_start_str = None
recorded_frames = 0

fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# =========================
# HELPER: SAVE LOG (APPEND)
# =========================
def save_log(start, end, duration, people):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    entry = {
        "event_id": f"evt_{len(data) + 1:03d}",
        "start_time": start,
        "end_time": end,
        "duration_sec": round(duration, 2),
        "people_count": people
    }

    data.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print("[LOG SAVED]", entry)

# =========================
# RECEIVE FRAME FROM PI
# =========================
@app.route("/upload", methods=["POST"])
def upload():
    global raw_frame, annotated_frame
    global intrusion_status, recording, video_writer, prev_intrusion
    global intrusion_start_time, intrusion_start_str
    global recorded_frames, live_counts

    file = request.files.get("image")
    if file is None:
        return "No image", 400

    frame = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return "Invalid frame", 400

    # Store RAW frame (for Live page)
    with frame_lock:
        raw_frame = frame.copy()

    # Run YOLO (person only)
    results = model(frame, conf=0.4, classes=[0])
    annotated = frame.copy()

    person_count = 0
    person_in_roi = False

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_count += 1

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if ROI_X1 < cx < ROI_X2 and ROI_Y1 < cy < ROI_Y2:
            person_in_roi = True

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    live_counts["person"] = person_count
    intrusion_status = person_in_roi

    # Draw ROI
    roi_color = (0, 0, 255) if person_in_roi else (255, 0, 0)
    cv2.rectangle(
        annotated,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        roi_color,
        2
    )

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
        intrusion_start_time = time.time()
        intrusion_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recorded_frames = 0

        filename = f"recordings/intrusion_{intrusion_start_str.replace(':','-')}.mp4"
        video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            FPS,
            (frame.shape[1], frame.shape[0])
        )
        recording = True
        print("[RECORDING STARTED]")

    if recording and video_writer is not None:
        video_writer.write(frame)
        recorded_frames += 1

    if not person_in_roi and prev_intrusion and recording and video_writer is not None:
        recording = False
        video_writer.release()
        video_writer = None

        end_time = time.time()
        end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - intrusion_start_time

        if intrusion_start_str is not None:
            save_log(intrusion_start_str, end_str, duration, person_count)

        print(f"[RECORDING STOPPED] Frames written: {recorded_frames}")

    prev_intrusion = person_in_roi

    # Store AI frame (for Dashboard)
    with frame_lock:
        annotated_frame = annotated.copy()

    return "OK"

# =========================
# STREAMING (RAW / AI)
# =========================
def stream(frame_type="annotated"):
    while True:
        with frame_lock:
            frame = raw_frame if frame_type == "raw" else annotated_frame

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

@app.route("/video_ai")
def video_ai():
    return Response(
        stream("annotated"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_raw")
def video_raw():
    return Response(
        stream("raw"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# =========================
# ROUTES
# =========================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/events")
def events():
    return render_template("events.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/system")
def system():
    return render_template("system.html")

@app.route("/api/events")
def api_events():
    if not os.path.exists(LOG_FILE):
        return jsonify([])

    with open(LOG_FILE, "r") as f:
        return jsonify(json.load(f))

@app.route("/api/counts")
def api_counts():
    return jsonify({
        "person": live_counts["person"],
        "intrusion": intrusion_status
    })

# =========================
# START
# =========================
if __name__ == "__main__":
    print(f"🌐 Cloud Server running on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
