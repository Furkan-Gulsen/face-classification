# Importing required packages
from keras.models import load_model
import numpy as np
import cv2

genderModelPath = 'models/age_model_with_vgg16.h5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

# pre-trained model
modelFile = "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
configFile = "faceDetection/models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def detectFacesWithDNN(frame):
    # A neural network that really supports the input value
    size = (300, 300)

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
            box = dnnFaces[0, 0, i, 3:7] * np.array(
                [width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            resized = frame[y - 20:y1 + 30, x - 10:x1 + 10]
            try:
                frame_resize = cv2.resize(resized, (100, 100))
            except:
                continue

            frame_resize = frame_resize.astype("float32")
            frame_scaled = frame_resize / 255.0
            frame_reshape = np.reshape(frame_scaled, (1, 100, 100, 3))
            frame_vstack = np.vstack([frame_reshape])
            gender_prediction = genderClassifier.predict(frame_vstack)

    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (720,420))

    if not ret:
        break

    frame = detectFacesWithDNN(frame)
    cv2.imshow("Gender Classification", frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
