import sys
sys.path.insert(1, '/Users/newuser/Projects/facialdetection/FaceRecognition/custom')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from frs import FaceRecognitionSystem

def identifyPerson(face, 
                   box, 
                   facial_features):
   
    embedding = fr.faceEmbeddings(face, 
                                  face_locations=box, 
                                  facial_features=facial_features)[0]
    
    prediction = face_classifier.predict_proba(embedding.reshape(1,-1))
    probability = prediction.max()
    
    if probability >= 0.5:
        name = str(fr.db[np.argmax(prediction) + 1]) + " {:.2f}%".format(probability * 100)
    else:
        name = "Unknown"
    
    return name

print("""
**********************************
Loading Face Recognition System...""")

DB = "/Users/newuser/Projects/facialdetection/FaceRecognition/custom/data/db.pkl"
EMBEDDINGS = "/Users/newuser/Projects/facialdetection/FaceRecognition/custom/data/embeddings.pkl"

fr = FaceRecognitionSystem(160, 
                           DB, 
                           EMBEDDINGS)
face_classifier = fr.faceClassifier()

print("Loading Face Mask classifier...")

#load my mask recognition model
mask_classifier = load_model("models/xception")

webcam = cv2.VideoCapture(0) 

while webcam.isOpened():
    
    _,frame = webcam.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB color channels	
    
    face_loc, face_features = fr.detectFaces(img)
    
    if face_features:
            
        for box, feature in zip(face_loc,face_features):
                
            (startX, startY, width, height) = box
            endX = startX + width
            endY = startY + width
            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(img.shape[1] - 1, endX), min(img.shape[0] - 1, endY))
                
            face = img[startY-20:endY+20, startX-20:endX+20]
            face = cv2.resize(face, (229, 229))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = mask_classifier.predict(face)[0]
                
            if mask > withoutMask:
                label = "Mask: {:.2f}%".format(mask * 100)
                color = (0, 180, 0) 
            else:
                #label = "No Mask: {:.2f}%".format(withoutMask * 100)
                label = identifyPerson(img, 
                                       [(startX, startY, endX, endY)], 
                                       [feature])
                color = (0, 60, 255) 
                if label is "Unknown":
                    color = (255, 60, 0)            
                
            x, y, wid, hei = startX, startY, endX, endY
                
            # ******** TOP **********
            cv2.rectangle(img, 
                          (x, y - 20), # upper left
                          (wid, y), # bottom right
                          color, 
                          -1) # -1 argument makes a filled rectangle
            # ******** TEXT **********
            cv2.putText(img, 
                        label, # text to draw
                        (startX+2, startY - 10), # text start coordinates
                        cv2.FONT_HERSHEY_PLAIN, # font
                        0.8, # fontscale
                        (255,255,255), # color
                        1) # thickness
            # ******** FACE FRAME *********
            cv2.rectangle(img, 
                          (startX, startY), 
                          (endX, endY), 
                          color, 
                          2)
                
    else:
        continue

    cv2.imshow("COVID-19 Mask Classifier App",img[:,:,::-1])

    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()




