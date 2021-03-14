# Importing required packages
from keras.models import load_model
import numpy as np
import argparse
import dlib
import cv2


class DetectFace:
    def __init__(self, isVideoWriter=False, frame):
        self.isVideoWriter = isVideoWriter
        self.frame = frame

    # for DNN
    def shapePoints(self, shape):
        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    # for DLIB
    def rectPoints(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)


    def prepareFrame(self):
