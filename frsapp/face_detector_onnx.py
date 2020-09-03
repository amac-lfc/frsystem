import cv2
#import dlib
import numpy as np
from imutils import face_utils
from box_utils import *
import os
import onnx

import onnxruntime as ort

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input

from onnx_tf.backend import prepare

video_capture = cv2.VideoCapture(0)

onnx_path = 'util/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
#predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

mask_classifier = load_model(os.path.join("frsapp","models","xception"))
#shape_predictor = dlib.shape_predictor('FacialLandmarks/shape_predictor_5_face_landmarks.dat')
#fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

while True:
    ret, frame = video_capture.read()
    if frame is not None:
        h, w, _ = frame.shape

        # preprocess img acquired
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
        img = cv2.resize(img, (640, 480)) # resize
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            cropped_face = frame[y1:y2, x1:x2]
            resized_face = cv2.resize(cropped_face, (299, 299))
            face = img_to_array(resized_face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, no_mask) = mask_classifier.predict(face)[0]
                    
            if mask > no_mask:
                label = "Mask: {:.2f}%".format(mask * 100)
                color = (0, 180, 0) 
            else:
                label = "No Mask: {:.2f}%".format(no_mask * 100)
                color = (0, 60, 255)          
                    
            x, y, wid, hei = x1, y1, x2, y2
                    
            # ******** TOP **********
            cv2.rectangle(frame, 
                            (x, y - 20), # upper left
                            (wid, y), # bottom right
                            color, 
                            -1) # -1 argument makes a filled rectangle
            # ******** TEXT **********
            cv2.putText(frame, 
                            label, # text to draw
                            (x1+2, y1 - 10), # text start coordinates
                            cv2.FONT_HERSHEY_PLAIN, # font
                            0.8, # fontscale
                            (255,255,255), # color
                            1) # thickness
            # ******** FACE FRAME *********
            cv2.rectangle(frame, 
                            (x1, y1), 
                            (x2, y2), 
                            color, 
                            2)
            
            
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #shape = shape_predictor(gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
            #shape = face_utils.shape_to_np(shape)
            #for (x, y) in shape:
                #cv2.circle(frame, (x, y), 2, (80,18,236), -1)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
            # cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # text = f"Face: {round(probs[i], 2)}"
            # cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()