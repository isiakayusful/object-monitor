import cv2
import torch
from flask import Flask, Response, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv5 model (change 'yolov5s' to your model, e.g., yolov5m or custom weights)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to perform detection
def detect_objects(frame):
    # Perform inference
    results = model(frame)
    # Parse results
    detections = results.pandas().xyxy[0]  # Get bounding boxes
    for _, row in detections.iterrows():
        # Draw bounding box and label
        x1, y1, x2, y2, conf, cls, name = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']]
        label = f"{name} ({conf:.2f})"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# Video stream generator
def video_stream():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Perform detection
        frame = detect_objects(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# Flask route for video stream
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for web interface
@app.route('/')
def index():
    return render_template('index.html')  # HTML file to display the video stream

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
