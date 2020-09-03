tf = None
try:
  tf = __import__("tensorflow-gpu")
  tf.operation_that_requires_gpu()
except:
  tf = __import__("tensorflow")
  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from frsystem.frs import FaceRecognitionSystem


def maskRecognizer(frs):
    #load my mask recognition model
    mask_classifier = load_model(os.path.join("frsapp","models","xception"))

    webcam = cv2.VideoCapture(0) 

    while webcam.isOpened():
        
        _,frame = webcam.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB color channels	
        
        face_loc, face_features = frs.detectFaces(img)
        
        if face_features:
                
            for box, feature in zip(face_loc,face_features):
                    
                (startX, startY, width, height) = box
                endX = startX + width
                endY = startY + height
                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(img.shape[1] - 1, endX), min(img.shape[0] - 1, endY))
                    
                face = img[startY-20:endY+20, startX-20:endX+20]
                face = cv2.resize(face, (299, 299))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                (mask, no_mask) = mask_classifier.predict(face)[0]
                    
                if mask > no_mask:
                    label = "Mask: {:.2f}%".format(mask * 100)
                    color = (0, 180, 0) 
                else:
                    label = "No Mask: {:.2f}%".format(no_mask * 100)
                    color = (0, 60, 255)          
                    
                x, y, wid, hei = startX, startY, endX, endY
                    
                # ******** TOP **********
                cv2.rectangle(frame, 
                            (x, y - 20), # upper left
                            (wid, y), # bottom right
                            color, 
                            -1) # -1 argument makes a filled rectangle
                # ******** TEXT **********
                cv2.putText(frame, 
                            label, # text to draw
                            (startX+2, startY - 10), # text start coordinates
                            cv2.FONT_HERSHEY_PLAIN, # font
                            0.8, # fontscale
                            (255,255,255), # color
                            1) # thickness
                # ******** FACE FRAME *********
                cv2.rectangle(frame, 
                            (startX, startY), 
                            (endX, endY), 
                            color, 
                            2)
                    
        else:
            continue

        cv2.imshow("COVID-19 Mask Classifier App", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    frs = FaceRecognitionSystem()
    
    maskRecognizer(frs)




