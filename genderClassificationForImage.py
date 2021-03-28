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


genderModelPath = 'models\genderModel_VGG16.hdf5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

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

# pre-trained model
modelFile = "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
configFile = "faceDetection/models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def detectFacesWithDNN(frame):
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
            resized = frame[y:y1, x:x1]
            try:
                frame_resize = cv2.resize(resized, genderTargetSize)
            except:
                continue

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
                color = genders[gender_label]["color"]
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                cv2.line(frame, (x, y1), (x + 20, y1 + 20), color, thickness=2)
                cv2.rectangle(frame, (x + 20, y1 + 20), (x + 135, y1 + 40),
                              color, -1)
                cv2.putText(frame, gender_result, (x + 25, y1 + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
    return frame


photo = cv2.imread("images/barackObama.jpeg")
photo = cv2.resize(photo, (1080, 720))  # option
frame = detectFacesWithDNN(photo)
cv2.imshow("Gender Classification", frame)
cv2.waitKey(0)
