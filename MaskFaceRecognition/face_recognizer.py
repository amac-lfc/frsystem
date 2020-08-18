import sys
sys.path.insert(1, '/Users/newuser/Projects/facialdetection/frsystem')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='3'

import cv2
from  numpy import argmin
from frsystem.frs import FaceRecognitionSystem
from frsystem.helper import getEmbeddingsList

def drawDetections(db,
                   frame, 
                   face_locations, 
                   face_names):
    # BGR format
    RED = (17, 69, 235)
    GREEN = (156, 234, 124)
    BLUE = (248, 99, 88)
    WHITE = (255, 255, 255)
    
    for (x, y, w, h), name in zip(face_locations, face_names):
        
        cv2.rectangle(frame,
                      (x, y),
                      (x+w, y+h), 
                      BLUE if name != "Unknown" else RED, 
                      thickness=3) # draw rectangle around face
					
		# Draw a label with a name below the face
        cv2.rectangle(frame,
                      (x, y+h),
                      (x+w, y+h+30),
                      BLUE if name != "Unknown" else RED,
                      cv2.FILLED)

		
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,
                    db[name] if name != "Unknown" else name,
                    (x + 6, y+h+25),
                    font,
                    0.8,
                    WHITE, 
                    1)
         
	
    return frame
							
def faceRecognizer(names_dict, embeddings_dict):

    fr = FaceRecognitionSystem(160, db_file=names_dict, embeddings_file=embeddings_dict)

    known_face_embeddings, known_face_names = getEmbeddingsList(fr.embeddings)
    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():
		# Grab a single frame of video
        _,frame = webcam.read()
		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# Only process every other frame of video to save time
        # Find all the faces and face embeddings in the current frame of video
        face_locations, facial_features = fr.detectFaces(rgb_small_frame)

        if len(face_locations) == 0:
            continue

        face_embeddings = fr.faceEmbeddings(rgb_small_frame, 
                                            face_locations=face_locations, 
                                            facial_features=facial_features)

        if face_embeddings.size != 0:
            
            face_names = []
            for face_embedding in face_embeddings:
                # See if the face is a match for the known face(s)
                face_distances = FaceRecognitionSystem.faceDistance(face_embedding, known_face_embeddings)
                matches = FaceRecognitionSystem.compareFaces(face_embedding, 
                                                             known_face_embeddings, 
                                                             distances=face_distances)
    
                best_match_index = argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"
                face_names.append(name)

        processed_frame = drawDetections(fr.db, 
                                         frame, 
                                         face_locations, 
                                         face_names)

        cv2.imshow("Face Recognizer", processed_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
   
    DB = "../FaceRecognitionSystem/data/db.pkl" # Path to serialized dictionary of id : name pairs of known faces
    EMBEDDINGS = "../FaceRecognitionSystem/data/embeddings.pkl"  # Path to serialized dictionary of id : faceEmbeddings of known faces
    # If using vggface model uncomment the two lines below
    # DB = "data/db_vggface.pkl"
    # EMBEDDINGS = "data/embeddings_vggface.pkl"    
    faceRecognizer(DB, EMBEDDINGS)