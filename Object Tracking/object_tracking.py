import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 10:
        break

cap.release()
cv2.destroyAllWindows()
