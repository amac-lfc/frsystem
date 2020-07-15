# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

import imutils
import time
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd

from PIL import ExifTags, Image
from collections import defaultdict
from mtcnn.mtcnn import MTCNN
from vggface import VGGFace
from keras.applications.imagenet_utils import preprocess_input
from algorithms import findCosineScore
from faceAligner import faceAligner
from faceDetector import isMatch

# initialize the video stream, then allow the camera sensor to warm up

def videoDetector():
  print("[INFO] starting video stream...")
  vs = VideoStream(src=0).start()
  time.sleep(2.0)
  # start the FPS throughput estimator
  fps = FPS().start()
  # loop over frames from the video file stream
  try:
    df = pd.read_csv("/Users/newuser/Projects/facialdetection/Identity-Match/known_faces.csv")
  except FileNotFoundError:
    print("known_faces.csv not found.")

  known_embeddings = df.to_dict('r')

  while True:
      # grab the frame from the threaded video stream
      frame = vs.read()
      # resize the frame to have a width of 600 pixels (while
      # maintaining the aspect ratio), and then grab the image
      # dimensions
      frame = imutils.resize(frame, width=600)
      (h, w) = frame.shape[:2]
      
      # apply OpenCV's deep learning-based face detector to localize
      # faces in the input image
      embedding_dict = getFaceEmbeddings(frame)
      #print(embedding_dict.keys())
    # loop over the detections
      for i in range(0, len(embedding_dict["name"])):
        box = np.array(embedding_dict["boundbox"][i][0])
        #print(box)
        (startX, startY, endX, endY) = box.astype("int")
        endX = endX - startX
        endY = endY - startY
          
        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
          continue

        for j in range(0, len(known_embeddings)):

          (is_match, info_string) = isMatch(known_embeddings[j]["Embedding"], embedding_dict["Embedding"][i])
            # draw the bounding box of the face along with the
            # associated probability
          if is_match:
            detected_name = known_embeddings[j]["Name"]

            if detected_name in embedding_dict["name"]:
              continue
            break   
        else:
          detected_name = "Unknown"
              
        #boundbox = embedding_dict["boundbox"][i][0]
      #detected_names_dict["boundbox"].append(boundbox)
      #detected_names_dict["name"].append(detected_name)

        text = detected_name
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
      # update the FPS counter
      fps.update()

    # show the output frame
      cv2.imshow("Frame", frame)
      key = cv2.waitKey(1) & 0xFF
      # if the `q` key was pressed, break from the loop
      if key == ord("q"):
          break
  # stop the timer and display FPS information
  fps.stop()
  print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
  # do a bit of cleanup
  cv2.destroyAllWindows()
  vs.stop()

def detectFaces(pixels, required_size=(224, 224)):
    """
    This function detects faces on a given image

    ## Args:
        filename ([type]): [description]
        required_size (tuple, optional): [description]. Defaults to (224, 224).

    ## Returns:
        dict: dictionary of extracted faces, each face has a nested dictionary consisting of two lists (face bounding box and face array of pixels)
    """ 

    # Dictionary of extracted faces
    extracted_faces = defaultdict(dict)

	# create the detector, using default weights
    detector = MTCNN()
	# detect faces in the image
    results = detector.detect_faces(pixels)
	
    # extract the bounding boxes of detected faces
    for i in range(len(results)):

        x1, y1, width, height = results[i]['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = pixels[y1:y2, x1:x2]

        # plt.imshow(face)
        # plt.show()

        # align the face for better precision
        landmarks = results[i]["keypoints"]
        face = faceAligner(pixels, landmarks, results[i]['box'])
        
        # plt.imshow(face)
        # plt.show()
        
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)

        # create nested dictionary for each face
        extracted_faces["face" + str(i+1)] = defaultdict()

        # append face array to the nested dictionary's "face_array" 
        extracted_faces["face" + str(i+1)]["face_array"] = list()
        extracted_faces["face" + str(i+1)]["face_array"].append(face_array)

        # append bound box coordinates to the nested dictionary's "bound_box"
        extracted_faces["face" + str(i+1)]["bound_box"] = list()
        extracted_faces["face" + str(i+1)]["bound_box"].append((x1, y1, x2, y2))

    return extracted_faces

def getFaceEmbeddings(image):
    """
    This function extracts face embeddings using VGGFace Architecture

    ## Args:
        filename (string): path to image 

    ## Returns:
        dict: a dictionary of extracted face embeddings (keys: name, embedding, boundbox)
    """
    
    # Create a dictionary of face embeddings
    face_embeddings_dict = {"name" : [], "embedding" : [], "boundbox" : [] }

	  # extract faces from an image 
    faces_dict = detectFaces(image)

    # instantiate model for obtaining face embeddings
    model = VGGFace()

    # loop over each face in our detected faces dictionary ( i.e face1, face2, face3 etc.)
    for face in faces_dict.keys():

        # convert face array into a numpy array
        sample = np.asarray(faces_dict[face]["face_array"], 'float32')

        # preprocess face array for inputting into the model
        sample = preprocess_input(sample)

        # predict face embedding
        face_embedding = model.predict(sample)

        # extract file name
        #name = ntpath.basename(filename).replace('-', ' ')[: -4].title()
        name = "unknown"

        # append name to the list of names in our dictionary
        face_embeddings_dict["name"].append(name)

        # append face embedding to the list of embeddings in our dictionary
        face_embeddings_dict["embedding"].append(face_embedding)

        # append bound box to the list of bound boxes in our dictionary
        boundbox = faces_dict[face]["bound_box"]
        face_embeddings_dict["boundbox"].append(boundbox)
        
    return face_embeddings_dict

videoDetector()