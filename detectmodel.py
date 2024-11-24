from modelcreation import model

def detect_objects(box):
    # Perform inference
    results = model(box)
    # Parse results
    detections = results.pandas().xyxy[0]  # Get bounding boxes
    for _, row in detections.iterrows():
        # Draw bounding box and label
        x1, y1, x2, y2, conf, cls, name = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']]
        label = f"{name} ({conf:.2f})"
        cv2.rectangle(box, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(box, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return box