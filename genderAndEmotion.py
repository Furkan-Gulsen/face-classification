# Importing required packages
from keras.models import load_model
import numpy as np
import cv2

genderModelPath = 'models\genderModel_VGG16.hdf5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

emotionModelPath = 'models/emotionModel.hdf5'
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}

genders = {
    0: {
        "label": "Female",
        "color": (245, 215, 130)
    },
    1: {
        "label": "Male",
        "color": (148, 181, 192)
    },
}

modelFile = "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "faceDetection/models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def predictGender(frame, resized, x, y, x1, y1):
    frame_resize = cv2.resize(resized, genderTargetSize)
    frame_resize = frame_resize.astype("float32")
    frame_scaled = frame_resize / 255.0
    frame_reshape = np.reshape(frame_scaled, (1, 100, 100, 3))
    frame_vstack = np.vstack([frame_reshape])
    gender_prediction = genderClassifier.predict(frame_vstack)
    gender_probability = np.max(gender_prediction)
    color = (255, 255, 255)
    if (gender_probability > 0.4):
        gender_label = np.argmax(gender_prediction)
        gender_result = genders[gender_label]["label"]
        return gender_result


def predictEmotion(frame, resized, x, h):
    grayFace = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grayFace = cv2.resize(grayFace, (emotionTargetSize))
    grayFace = grayFace.astype('float32')
    grayFace = grayFace / 255.0
    grayFace = (grayFace - 0.5) * 2.0
    grayFace = np.expand_dims(grayFace, 0)
    grayFace = np.expand_dims(grayFace, -1)
    emotion_prediction = emotionClassifier.predict(grayFace)
    emotion_probability = np.max(emotion_prediction)
    if (emotion_probability > 0.36):
        emotion_label_arg = np.argmax(emotion_prediction)
        return emotions[emotion_label_arg]['emotion']


def predictGenderAndEmotion(frame):
    size = (300, 300)
    scalefactor = 1.0
    swapRB = (104.0, 117.0, 123.0)
    height, width = frame.shape[:2]
    resizedFrame = cv2.resize(frame, size)
    blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
    net.setInput(blob)
    dnnFaces = net.forward()
    for i in range(dnnFaces.shape[2]):
        confidence = dnnFaces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnnFaces[0, 0, i, 3:7] * np.array(
                [width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            color = (245, 215, 130)
            resized = frame[y - 20:y1 + 30, x - 10:x1 + 10]
            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            try:
                gender_result = predictGender(frame, resized, x, y, x1, y1)
                emotion_result = predictEmotion(frame, resized, x, y1)
            except:
                continue

            cv2.rectangle(frame, (x + 20, y1 + 20), (x + 170, y1 + 55), color,
                          -1)
            cv2.line(frame, (x, y1), (x + 20, y1 + 20), color, thickness=2)
            cv2.putText(frame, '{}/{}'.format(gender_result, emotion_result),
                        (x + 25, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

    return frame


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = predictGenderAndEmotion(frame)
    cv2.imshow("Gender Classification", frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
