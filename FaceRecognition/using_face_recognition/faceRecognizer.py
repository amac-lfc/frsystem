import face_recognition
import cv2
import numpy as np
import pickle

def connectToDB():

	f = open("db.pkl","rb")
	db = pickle.load(f)         #ref_dict=ref vs name
	f.close()

	f = open("embeddings.pkl","rb")
	embeddings = pickle.load(f)      #embed_dict- ref  vs embedding 
	f.close()

	return db, embeddings

def displayDetections(db, frame, face_locations, face_names):
	
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		red = (235, 69, 17)
		green = (124, 234, 156)
		blue = (88, 99, 248)
		
		cv2.rectangle(frame, 
					(left, top), 
					(right, bottom), 
					blue if name == "Unknown" else red, 2) # draw rectangle around face
					
		# Draw a label with a name below the face
		cv2.rectangle(frame, 
					(left, bottom - 35), 
					(right, bottom), 
					blue if name == "Unknown" else red, 
					cv2.FILLED)

		
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, 
					db[name] if name != "Unknown" else "Unknown", 
					(left + 6, bottom - 6), 
					font, 1.0, 
					(255, 255, 255), 1)
	
	return frame
							
def main():

	db, embeddings = connectToDB()

	############################################################################  encodings and ref_ids 
	known_face_encodings = []  # encodings of faces
	known_face_names = []	   # ref_id of faces

	for ref_id , embed_list in embeddings.items():
		for embed in embed_list:
			known_face_encodings +=[embed]
			known_face_names += [ref_id]
	
	#############################################################frame capturing from camera and face recognition
	webcam = cv2.VideoCapture(0)
	
	face_locations = []
	face_encodings = []
	face_names = []
	process_this_frame = True

	while True  :
		# Grab a single frame of video
		ret, frame = webcam.read()

		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]

		# Only process every other frame of video to save time
		if process_this_frame:
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame)
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

			face_names = []

			for face_encoding in face_encodings:
				# See if the face is a match for the known face(s)
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)

				if matches[best_match_index]:
					name = known_face_names[best_match_index]
				else:
					name = "Unknown"
				
				face_names.append(name)

		process_this_frame = not process_this_frame


		frame = displayDetections(db, frame, face_locations, face_names)
		
		cv2.imshow('Video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	webcam.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()