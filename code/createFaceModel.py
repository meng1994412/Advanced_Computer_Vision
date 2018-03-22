import cv2
import argparse
import imutils
import numpy as np
import os
from PIL import Image


cascadePath = "haar_cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(path):
    '''
    This function:
            Convert images into matrices
            Assign a label to each image according the person
            Using the data set from image to train the machine

    Inputs:
            path: path to image directory

    Returns:
            matrix of images, labels
    '''

    image_paths = [os.path.join(path, file)
                   for file in os.listdir(path) if not file.endswith('.sad')]
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        image = imutils.resize(image, width = min(500, image.shape[1]))
        nbr = int(os.path.split(image_path)[1].split(
           ".")[0].replace("subject", ""))
        #nbr = eval(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Add faces for training", image[y: y + h, x: x + w])
            cv2.imshow('window', image[y: y + h, x : x + w])
            cv2.waitKey(50)
    return images, labels


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "path to image directory")
args = vars(ap.parse_args())
path = args["images"]
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))
'''
save the trained dataset to cont.yaml file
'''
recognizer.write("model.yaml")

