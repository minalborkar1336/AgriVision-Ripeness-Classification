
import cv2
import numpy as np
from ultralytics import YOLO

# Load the pretrained object detector and your trained classifier
detector = YOLO("yolov8n.pt")
classifier = YOLO("best.pt")

# COCO class ID for banana
BANANA_CLASS_ID = 46

# Detection confidence threshold
DETECT_THRESHOLD = 0.35

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Banana Ripeness Detection started. Press Q to quit.")

class_names = classifier.names

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Detect objects in the webcam frame
    detections = detector.predict(
        frame,
        imgsz=640,
        conf=DETECT_THRESHOLD,
        verbose=False
    )[0]

    banana_found = False

    for box in detections.boxes:
        detected_class = int(box.cls[0])

        if detected_class != BANANA_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Crop the detected banana
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        banana_found = True

        # Classify only the banana crop
        prediction = classifier.predict(
            crop,
            imgsz=224,
            verbose=False
        )[0].probs

        probabilities = prediction.data.cpu().numpy()
        best_id = int(np.argmax(probabilities))
        best_conf = float(probabilities[best_id]) * 100
        best_name = class_names[best_id]

        # Highlight banana with a green bounding box
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Banana: {best_name} ({best_conf:.1f}%)",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # Display all four class probabilities
        panel_x = 15
        panel_y = 55
        line_height = 32

        for class_id, probability in enumerate(probabilities):
            name = class_names[class_id]
            percent = float(probability) * 100
            is_best = class_id == best_id

            color = (0, 255, 0) if is_best else (230, 230, 230)

            text = f"{name}: {percent:.1f}%"
            if is_best:
                text = ">> " + text + "  BEST"

            cv2.putText(
                frame,
                text,
                (panel_x, panel_y + class_id * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2 if is_best else 1
            )

    if not banana_found:
        cv2.putText(
            frame,
            "No banana detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("Banana Ripeness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
