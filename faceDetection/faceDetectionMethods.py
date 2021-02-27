# Importing required packages
import numpy as np
import cv2


def detectFacesWithDNN(frame):
    # pre-trained model
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel" 
    # prototxt has the information of where the training data is located.
    configFile = "models/deploy.prototxt" 
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

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
            box = dnnFaces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1),  (193, 69, 42), 2)
    return frame
