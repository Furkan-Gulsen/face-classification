# Importing required packages
from keras.models import load_model
import numpy as np
import dlib
import cv2


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

emotionModelPath = 'models/age_prediction_with_cnn.h5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

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
            grayFace = cv2.resize(grayFace, (48, 48))
        except:
            continue

        grayFace = grayFace.reshape(-1, 48, 48, 1)
        emotion_prediction = emotionClassifier.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)
        if (emotion_probability > 0.45):
            emotion_label_arg = np.argmax(emotion_prediction)
            color = (100, 100, 100)
            cv2.putText(frame, str(emotion_label_arg), (x + 25, y + h + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
