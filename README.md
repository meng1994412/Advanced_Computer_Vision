# Human Tracking, Face Detection and Recognition
## Overview
Nowadays, many closedcircuit television cameras are applied in the public places such as bank, hotel, and etc. for the safety purposes. Though people do not have to monitor the area with the help of surveillance cameras, they still have to spend time trying to identify someone in the video. And sometimes this could cost a huge amount of time to search by people and results are not good. If a computer could identify a person by itself, it will save lots of human efforts.

In this project, a program is developed which can detect a person and his/her face in a given video and then recognize him/her by using features in the pre-trained model of that person. If the person is detected in the video for the first time, a label number will be assigned to that person. And that person can be tracked and recognized in any other given videos.

## Objectives
* develop a program that can detect a person and his/her face in a given video, and then recognize him/her by using features in the pre-trained model of that person.

## Requirements
* Python3 (3.6)
* OpenCV  (3.3)
* Python libraries:
  - Numpy
  - Imutils
  - Python Image Library (PIL)
  
## Algorithms & Methods Involved
* Haar feature & cascade classifiers
* Local Binary Pattern Histogram (LBPH)
* Histogram of Gradients (HOG)
* Support Vector Machine (SVM)
* Non-maximum Suppresion

## Approaches
* The program follow the steps shown below:
  1. First it reads an input video each frame one by one.
  2. For each frame, it tries to detect human body. If a body is detected, it will draw a red rectangle encircled the body. 
  3. Then it tries to detect human face.
  4. If the face is detected, it will try to recognize it based on features in pre-trained models.
  5. If the face is recognized, it will put a label above the face. Otherwise it repeats the step 2 to next frame.
* The instruction of the repo
  1. `code/`: This directory contains [code](https://github.com/meng1994412/EECS432_Advanced_Computer_Vision/tree/master/code):
     - `main.py`: This is the main python file that detects and recognizes human.
     - `extractFace.py`: This is data extraction file that takes 600 frontal face photos of each subject.
     - `createFaceModel.py`: This is model creation file that create a LBPH feature model based on the photos extracted from `extractFace.py`.
  2. `data/`: This [directory](https://github.com/meng1994412/EECS432_Advanced_Computer_Vision/tree/master/data) contains some data samples (not all) extracted from `extractFace.py`.
  3. `haar_cascade/`: This [directiory](https://github.com/meng1994412/EECS432_Advanced_Computer_Vision/tree/master/haar_cascades) contains the pre-trained classifiers for frontal face detection used in `main.py`, `extractFace.py`, and `createFaceModel.py`, which is obtained from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)
  4. `videos/`: This [directory](https://github.com/meng1994412/EECS432_Advanced_Computer_Vision/tree/master/video) contains videos for testing.
  
 ## Proceeding code
 * To run the python code you have to put all the data images or input videos in one folder and then provide the path of that folder as command line argument. For example:
 ```python3 createFaceModel.py -i data```
 This command line will run python file `createFaceModel.py` and take images from folder named data to create feature model.
 ```python3 main.py -v videos```
 This command line will run python file `main.py` and take videos from folder named video to create output.
