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


faceLandmarks = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

genderModelPath = 'models/genderModel.hdf5'
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

cap = cv2.VideoCapture(0)
label = {0:"female",1:"male"} 

while True:
    ret, frame = cap.read()

    if not ret:
        break


    rects = detector(frame, 0)
    for rect in rects:
    	shape = predictor(frame, rect)
    	points = shapePoints(shape)
    	(x, y, w, h) = rectPoints(rect)
    	resized = frame[y: y+h, x:x+w]
    	try:
    		frame_resize = cv2.resize(resized, genderTargetSize)
    	except:
    		continue
    	frame_resize = frame_resize.astype('float32')
    	frame_scaled = frame_resize/255.0
    	reshape = np.reshape(frame_scaled,(1, 150, 150 ,3))
    	img = np.vstack([reshape])
    	gender_prediction = emotionClassifier.predict(img)
    	gender_probability = np.max(gender_prediction)
    	gender_label = np.argmax(gender_prediction)
    	cv2.putText(frame, label[gender_label] + "("+ str(gender_probability) + ")" , (x+5, y+h-5),
    		cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255), 1, cv2.LINE_AA)
    	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255,255), 2)
    
    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
    	break


cap.release()
cv2.destroyAllWindows()
