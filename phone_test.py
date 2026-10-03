import cv2
from ultralytics import YOLO

# YOLO model load
model = YOLO("yolo11n.pt")

# Webcam start
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera open nahi hua!")
        break

    # YOLO detection
    results = model(frame, verbose=False)

    # Detection result draw
    annotated_frame = results[0].plot()

    # Screen par show
    cv2.imshow("YOLO Phone Detection", annotated_frame)

    # Q dabane par exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Camera release
cap.release()
cv2.destroyAllWindows()