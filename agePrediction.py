# Importing required packages
from keras.models import load_model
import numpy as np
import dlib
import cv2

prediction_labels = [
    "generation z", "generation y", "generation x", "baby boomers",
    "the silent generation"
]


def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

ageModelPath = 'models/age_model_with_cnn.h5'
agePrediction = load_model(ageModelPath, compile=False)
ageTargetSize = agePrediction.input_shape[1:3]

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))

    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, (ageTargetSize))
        except:
            continue

        grayFace = grayFace.reshape(-1, 48, 48, 1)
        age_prediction = agePrediction.predict(grayFace)
        age_probability = np.max(age_prediction)
        color = (23, 164, 28)

        if (age_probability > 0.45):
            prediction = np.argmax(age_prediction)
            result = prediction_labels[prediction]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (color), 2)
            cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                     color,
                     thickness=2)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 135, y + h + 40),
                          color, -1)
            cv2.putText(frame, result, (x + 25, y + h + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        else:
            color = (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Age Prediction", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
