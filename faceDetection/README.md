<h1>Face Detection Methods</h1>

Today, there are many different face detection methods. While some of these are very good detections, some can make bad detections. Causes of bad detections:
- The person wears glasses,
- The person is in a dim environment,
- The person's entire face is not visible
and because of similar reasons, the detection rate of these methods also changes.

I compared the four most used face detection methods in the world. These:
- DNN
- DLIB
- Haarcascade
- MTCNN

## DNN
If you want to run face detection with DNN:
```python
python faceDetection.py -m DNN
```
<img src="https://github.com/Furkan-Gulsen/face-classification/blob/main/faceDetection/outputs/dnn_output.gif?raw=true" />
 
</br>

## DLIB
If you want to run face detection with DLIB:
```python
python faceDetection.py -m DLIB
```
<img src="https://github.com/Furkan-Gulsen/face-classification/blob/main/faceDetection/outputs/dlib_output.gif?raw=true" />

</br>


## Haarcascade
If you want to run face detection with Haarcascade:
```python
python faceDetection.py -m HAARCASCADE
```
<img src="https://github.com/Furkan-Gulsen/face-classification/blob/main/faceDetection/outputs/haarcascade_output.gif?raw=true" />

</br>

## MTCNN
If you want to run face detection with MTCNN:
```python
python faceDetection.py -m MTCNN
```
<img src="https://github.com/Furkan-Gulsen/face-classification/blob/main/faceDetection/outputs/mtcnn_output.gif?raw=true" />
