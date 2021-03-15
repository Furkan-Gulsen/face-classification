# Importing required packages
import numpy as np
import cv2
from faceProcess import FaceProcess

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = FaceProcess(frame).run()

    cv2.imshow("frame", frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
