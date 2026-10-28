from ultralytics import YOLO
import cv2

MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.25

model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open camera. Check camera permissions or try another CAMERA_INDEX."
    )

print("Live tomato detection started. Press Q to quit.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read a frame from the camera.")
        break

    results = model.predict(frame, conf=CONFIDENCE, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Tomato Freshness Detection", annotated)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

camera.release()
cv2.destroyAllWindows()
