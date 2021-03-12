<h1 align="center">Face Classification</h1>
<p align="center">
  <img align="center" src="https://github.com/Furkan-Gulsen/face-classification/blob/main/assets/faces.jpg" width="100%"/>
</p>

<p align="center">
  In this project, one or more human faces are detected in real time and predictions are made about the faces detected by AI models trained in the background.
<p>

---

<h2> Emotion Recognition </h2>
<img src="https://github.com/Furkan-Gulsen/face-classification/blob/main/outputs/EmotionRecognitionOutput.gif?raw=true" />
<p>
  I used only Transfer Learning models in this project to get faster and better results.
  For Emotion Recognation:
</p>
<ul>
  <li>VGG16,</li>
  <li>VGG19,</li>
  <li>ResNet,</li>
  <li>Inception,</li>
  <li>Xception,</li>
</ul>
<p>
  I experimented with transfer learning models. I used the Xception model in this section because it gives the best result.
</p>

If you want to run real time face emotion recognition:
```python
python emotionRecognitionWithDNN.py
```
or
```python
python emotionRecognitionWithDLIB.py
```
