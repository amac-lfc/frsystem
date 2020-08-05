
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='3'


import cv2
from  numpy import argmin
from FaceRecognitionSystem import FaceRecognitionSystem
from time import sleep

DB = "/Users/newuser/Projects/facialdetection/FaceRecognition/custom/data/db.pkl"
EMBEDDINGS = "/Users/newuser/Projects/facialdetection/FaceRecognition/custom/data/embeddings.pkl"

fr = FaceRecognitionSystem(160, DB, EMBEDDINGS)
db = fr.db

RED = (235, 69, 17)
GREEN = (124, 234, 156)
BLUE = (88, 99, 248)
WHITE = (255, 255, 255)

def displayDetections(frame, face_locations, face_names):
    
    for (x, y, w, h), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
        x *= 4
        y *= 4
        w *= 4
        h *= 4
        
        cv2.rectangle(frame, 
					(x, y), 
					(x+w, y+h), 
					BLUE if name == "Unknown" else RED, thickness=3) # draw rectangle around face
					
		# Draw a label with a name below the face
        cv2.rectangle(frame, 
					(x, y+h), 
					(x+w, y+h+30), 
					BLUE if name == "Unknown" else RED, cv2.FILLED)

		
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, 
					db[name] if name != "Unknown" else name, 
					(x + 6, y+h+25), 
					font, 0.7, 
					WHITE, 1)
         
	
    cv2.imshow("Video", frame)
							
def faceRecognizer():

    known_face_embeddings, known_face_names = fr.getEmbeddingsList()
    webcam = cv2.VideoCapture(0)
    #sleep(2)
    face_locations = []
    face_embeddings = []
    face_names = []

    process_this_frame = True

    while webcam.isOpened():
		# Grab a single frame of video
        _,frame = webcam.read()
		# Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
		# Only process every other frame of video to save time
        if process_this_frame:
			# Find all the faces and face embeddings in the current frame of video
            face_locations, facial_features = fr.detectFaces(rgb_small_frame)

            if len(face_locations) == 0:
                continue

            face_embeddings = fr.faceEmbeddings(rgb_small_frame, face_locations=face_locations, facial_features=facial_features)

            face_names = []

            for face_embedding in face_embeddings:
				# See if the face is a match for the known face(s)
                face_distances = fr.faceDistance(face_embedding, face_embeddings=known_face_embeddings)
                matches = fr.compareFaces(face_embedding, known_face_embeddings=known_face_embeddings, distances=face_distances)
    
                best_match_index = argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"
                face_names.append(name)

        process_this_frame = not process_this_frame

        displayDetections(frame, face_locations, face_names)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    faceRecognizer()