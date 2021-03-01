# Importing required packages
import models.faceDetectionMethods as fdm
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", type=str, help="choose a face detection pattern")
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

# check method
if args["method"] == None:
    raise ValueError("There is no such method. Please choose a method.")


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

    if args["method"].lower() == "dnn":
        frame = fdm.detectFacesWithDNN(frame)
    elif args["method"].lower() == "dlib":
        frame = fdm.detectFacesWithDLIB(frame)
    elif args["method"].lower() == "haarcascade":
        frame = fdm.detectFacesWithCascade(frame)
    elif args["method"].lower() == "mtcnn":
        frame = fdm.detectFacesWithMTCNN(frame)
    else:
        raise ValueError("There is no such method. Please check the models")
    

    if args["isVideoWriter"] == True:
        videoWrite.write(frame)

    cv2.imshow("Face Detection Model", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
