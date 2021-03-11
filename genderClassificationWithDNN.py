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


# pre-trained model
modelFile = "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
configFile = "faceDetection/models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detectFacesWithDNN(frame):
    # A neural network that really supports the input value
    size = (300,300)

    # After executing the average reduction, the image needs to be scaled
    scalefactor = 1.0

    # These are our mean subtraction values. They can be a 3-tuple of the RGB means or
    # they can be a single value in which case the supplied value is subtracted from every
    # channel of the image.
    swapRB = (104.0, 117.0, 123.0)

    height, width = frame.shape[:2]
    resizedFrame = cv2.resize(frame, size)
    blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
    net.setInput(blob)
    dnnFaces = net.forward()
    for i in range(dnnFaces.shape[2]):
        confidence = dnnFaces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnnFaces[0, 0, i, 3:7] * np.array([width-20, height+20, width-20, height+20])
            (x, y, x1, y1) = box.astype("float32")
            cv2.rectangle(frame, (x, y), (x1, y1), (193, 69, 42), 2)
    return frame


genderModelPath = 'models\genderModel_VGG16.hdf5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

genders =  {
    0: { "label": "Female", "color": (245,215,130) },
    1: { "label": "Male", "color": (148,181,192) },
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = detectFacesWithDNN(frame)

    cv2.imshow("Gender Classification", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
    	break


cap.release()
cv2.destroyAllWindows()
