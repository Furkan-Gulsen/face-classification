# Importing required packages
from keras.models import load_model
import numpy as np
import cv2

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
                cv2.rectangle(frame, (x + 20, y1 + 20), (x + 130, y1 + 55),
                              color, -1)
                cv2.line(frame, (x, y1), (x + 20, y1 + 20), color, thickness=2)
                cv2.putText(frame, gender_result, (x + 25, y1 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            else:
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
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
