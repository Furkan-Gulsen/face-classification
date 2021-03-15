# Importing required packages
from keras.models import load_model
import numpy as np
import cv2

genderModelPath = 'models\genderModel_VGG16.hdf5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]


class GenderClassification:
    def __init__(self, face, x, y, x1, y1):
        self.face = face
        self.x = x
        self.y = y
        self.x1 = x1
        self.y1 = y1

    @staticmethod
    def getEmotion(index):
        return {
            0: {
                "label": "Female",
                "color": (245, 215, 130)
            },
            1: {
                "label": "Male",
                "color": (148, 181, 192)
            }
        }[index]

    def resizeFace(self):
        frame_resize = cv2.resize(self.face, genderTargetSize)
        frame_resize = frame_resize.astype("float32")
        frame_scaled = frame_resize / 255.0
        frame_reshape = np.reshape(frame_scaled, (1, 100, 100, 3))
        frame_vstack = np.vstack([frame_reshape])
        return frame_vstack

    def predict(self):
        resizedFace = self.resizeFace()
        gender_prediction = genderClassifier.predict(resizedFace)
        gender_probability = np.max(gender_prediction)
        if (gender_probability > 0.4):
            gender_label = np.argmax(gender_prediction)
            gender_result = self.getEmotion(gender_label)
            return gender_result
