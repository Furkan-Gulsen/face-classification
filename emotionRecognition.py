# Importing required packages
from keras.models import load_model
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", type=str, help="choose a face detection pattern")
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

emotion_offsets = (20, 40)
emotions =  {
    0: { "emotion": "Angry", "color": (193, 69, 42) },
    1: { "emotion": "Disgust", "color": (164, 175, 49) },
    2: { "emotion": "Fear", "color": (40, 52, 155) },
    3: { "emotion": "Happy", "color": (23, 164, 28) },
    4: { "emotion": "Sad", "color": (164, 93, 23) },
    5: { "emotion": "Suprise", "color": (218, 229, 97) },
    6: { "emotion": "Neutral", "color": (108, 72, 200) }
}
