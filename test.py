from flask import Flask, Response, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Load YOLO (Cloud model)
model = YOLO("yolov8n.pt")

latest_frame = None
lock = threading.Lock()

live_counts = {
    "person": 0
}

print("✅ YOLO Loaded on Cloud Server")

# -----------------------------
# Receive Frame from Raspberry Pi
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():

    global latest_frame, live_counts

    file = request.files.get("image")
    if file is None:
        return "No image", 400

    data = file.read()
    img = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Run YOLO (person only)
    results = model(frame, conf=0.4, classes=[0])
    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0

    annotated = results[0].plot()

    # Overlay count
    cv2.putText(
        annotated,
        f"Persons: {count}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    with lock:
        latest_frame = annotated.copy()
        live_counts["person"] = count

    return "OK"


# -----------------------------
# MJPEG Stream to Dashboard
# -----------------------------
def generate():

    global latest_frame

    while True:
        with lock:
            if latest_frame is None:
                continue

            ret, jpg = cv2.imencode(".jpg", latest_frame)
            if not ret:
                continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() +
            b"\r\n"
        )


@app.route("/video_ai")
def video_ai():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# -----------------------------
# API for Dashboard Counts
# -----------------------------
@app.route("/api/counts")
def api_counts():
    return jsonify(live_counts)


# -----------------------------
# Dashboard Page
# -----------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    print("🌐 Cloud Server running on port 8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
