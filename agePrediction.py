# Importing required packages
from keras.models import load_model
import numpy as np
import argparse
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

emotionModelPath = 'models/age_model_with_cnn.h5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]


def get_age(distr):
    distr = distr * 4
    if distr >= 0.65 and distr <= 1.4: return "0-18"
    if distr >= 1.65 and distr <= 2.4: return "19-30"
    if distr >= 2.65 and distr <= 3.4: return "31-80"
    if distr >= 3.65 and distr <= 4.4: return "80 +"
    return "Unknown"


def get_gender(prob):
    if prob < 0.5: return "Male"
    else: return "Female"


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
        grayFace = grayFrame[y - 20:y + h + 30, x - 10:x + w + 10]
        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue

        grayFace = grayFace.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotion_prediction = emotionClassifier.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)
        if (emotion_probability > 0.36):
            age = get_age(emotion_prediction[0])
            gender = get_gender(emotion_prediction[1])
            cv2.putText(
                frame,
                "Predicted Gender: {}, Predicted Age: {}".format(gender, age),
                (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (245, 215, 130), 1, cv2.LINE_AA)

    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
