import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
CROPED = '/home/arnur/facialdetection/datasets/cropped-faces/'
def detect_faces(cascade, test_image, scaleFactor = 1.1):

    face_crop = []

    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    to_crop = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 5)
        name = str(random.randint(1,50))
        cv2.imwrite(os.path.join(CROPED , name + 'cropped.png'), image_copy)
        face_crop.append(to_crop[y:y+h, x:x+w])

    return face_crop

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save(face, path, img_name):
  name = img_name[:-4] + '-crop.png'
  cv2.imwrite(os.path.join(path , name), face) 
  return name