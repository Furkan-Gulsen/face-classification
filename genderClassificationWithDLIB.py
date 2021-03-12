# Importing required packages
from keras.models import load_model
import numpy as np
import argparse
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

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


faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

genderModelPath = 'models\genderModel_VGG16.hdf5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

genders =  {
    0: { "label": "Female", "color": (245,215,130) },
    1: { "label": "Male", "color": (148,181,192) },
}

cap = cv2.VideoCapture(0)

if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M","J","P","G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22, (capWidth, capHeight))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rects = detector(frame, 0)
    for rect in rects:
        shape = predictor(frame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        resized = frame[y-20: y+h+30, x-10:x+w+10]
        cv2.imshow("resized: ", resized)
        try:
            frame_resize = cv2.resize(resized, genderTargetSize)
        except:
            continue

        frame_resize = frame_resize.astype('float32')
        frame_scaled = frame_resize/255.0
        frame_reshape = np.reshape(frame_scaled,(1, 100, 100 ,3))
        frame_vstack = np.vstack([frame_reshape])
        gender_prediction = genderClassifier.predict(frame_vstack)
        gender_probability = np.max(gender_prediction)
        color = (255,255,255)
        if(gender_probability > 0.6):
        	gender_label = np.argmax(gender_prediction)
        	gender_result = genders[gender_label]["label"]
        	color = genders[gender_label]["color"]
        	cv2.putText(frame, gender_result , (x+5, y+h-5),
        		cv2.FONT_HERSHEY_SIMPLEX, 1 , color, 2, cv2.LINE_AA)
        	cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
        	cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if args["isVideoWriter"] == True:
        videoWrite.write(frame)

    cv2.imshow("Gender Classification", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
    	break


cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
