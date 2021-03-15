# Importing required packages
import numpy as np
import cv2
from genderClassification import GenderClassification


class FaceProcess:
    frame_w = 720
    frame_h = 420

    def __init__(self, frame):
        self.frame = frame
        self.modelFile = "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "faceDetection/models/dnn/deploy.prototxt"
        self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)

    @property
    def frameSize(self):
        print("Width : {}\nHeight: {}".format(self.frame_w, self.frame_h))

    # new_size = [width, height]
    @frameSize.setter
    def frameSize(self, new_size):
        try:
            self.frame_w = new_size[0]
            self.frame_h = new_size[1]
            print("Frame size change has been done successfully")
        except:
            print("Make sure you enter the parameter in the format",
                  "[widtht <number>, height <number>]")

    def faces(self, frame):
        size = (300, 300)
        scalefactor = 1.0
        swapRB = (104.0, 117.0, 123.0)
        resizedFrame = cv2.resize(frame, size)
        blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
        self.net.setInput(blob)
        faces = self.net.forward()
        return faces

    def run(self):
        height, width = self.frame.shape[:2]
        _faces = self.faces(self.frame)
        for i in range(_faces.shape[2]):
            confidence = _faces[0, 0, i, 2]
            if confidence > 0.5:
                box = _faces[0, 0, i, 3:7] * np.array(
                    [width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                color = (245, 215, 130)
                resized = self.frame[y - 20:y1 + 30, x - 10:x1 + 10]
                cv2.rectangle(self.frame, (x, y), (x1, y1), color, 2)
                try:
                    gender_result = GenderClassification(
                        resized, x, y, x1, y1).predict()
                except:
                    continue

                cv2.rectangle(self.frame, (x + 20, y1 + 20),
                              (x + 170, y1 + 55), gender_result['color'], -1)
                cv2.line(self.frame, (x, y1), (x + 20, y1 + 20),
                         gender_result['color'],
                         thickness=2)
                cv2.putText(self.frame, '{}'.format(gender_result['label']),
                            (x + 25, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)

        return self.frame
