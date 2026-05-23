from ultralytics import YOLO
import os
import cv2

# Load pretrained YOLOv8 (nano for speed, but you can pick yolov8s/m/l/x)
model = YOLO(os.path.dirname(__file__)+"/data/yolov8fire_detection.pt")

# Open a webcam stream (0 = default camera, or replace with video path)
video = os.path.dirname(__file__)+"/data/small-fire.avi"
cap = cv2.VideoCapture(video)   # input source

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection (stream=True keeps results as generator for efficiency)
    results = model(frame, stream=True, classes=[0])  # class 2 = "car"

    # Loop through detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # confidence
            cls = int(box.cls[0])      # class id

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label with class and confidence
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLO Car Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()