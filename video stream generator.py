import cv2
import numpy as np

# Load the YOLOv5 model
net = cv2.dnn.readNet("yolov5s.onnx")  # Path to the ONNX model

# Load class names (COCO dataset)
classes = []
with open("coco.names", "r") as f:  # Path to class names file
    classes = [line.strip() for line in f.readlines()]

# Function to detect objects
def detect_objects(frame):
    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass
    outputs = net.forward()

    # Get dimensions
    height, width, _ = frame.shape

    # Parse outputs
    for output in outputs[0]:
        confidence = output[4]
        if confidence > 0.5:  # Confidence threshold
            scores = output[5:]
            class_id = np.argmax(scores)
            score = scores[class_id]
            if score > 0.5:
                # Get bounding box
                x, y, w, h = output[:4] * np.array([width, height, width, height])
                x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)

                # Draw bounding box and label
                label = f"{classes[class_id]}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Capture video
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    frame = detect_objects(frame)

    # Show the result
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
