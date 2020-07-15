import sys
import cv2 
import os
import face_recognition
import pickle
import uuid 
import ntpath
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def updateDatabase(name):
	#  Try to open database of names, if doesn't exist create one
	try:
		f = open("db.pkl", "rb")
		db = pickle.load(f)
		if name in db.values():
			ref_id = db.index(name)
		else:
			ref_id = uuid.uuid4().int
		f.close()
	except:
		db = {}
		ref_id = uuid.uuid4().int

	db[ref_id]=name

	f = open("db.pkl", "wb")
	pickle.dump(db, f)
	f.close()

	return db, ref_id

def readEmbeddingsFromDatabase():
	try:
		f = open("embeddings.pkl", "rb")
		embeddings = pickle.load(f)
		f.close()
	except:
		embeddings = {}

	return embeddings

def addEmbeddingsToDatabase(embeddings):
	
	f = open("embeddings.pkl", "wb")
	pickle.dump(embeddings, f)
	f.close()
	print("Embeddings added to database.")
	

def updateEmbeddings(image, ref_id, embeddings):

	face_encoding = face_recognition.face_encodings(image)[0]

	if ref_id in embeddings:
		embeddings[ref_id]+=[face_encoding]
	else:
		embeddings[ref_id]=[face_encoding]

	return embeddings


def addEmbeddingsFromCamera(ref_id, embeddings):

	how_many = int(input("How many embeddings you want to add?\n"))

	for _ in range(how_many):

		key = cv2.waitKey(1)
		webcam = cv2.VideoCapture(0)

		while True:
			check, frame = webcam.read()
			# print(check) # prints true as long as the webcam is running
			# print(frame) # prints matrix values of each frame
			cv2.imshow("Capturing", frame) # Display frame
			
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # Resize frame for better performance
			rgb_small_frame = small_frame[:, :, ::-1] # convert the frame from GBR (OpenCV) to RGB 
			
			key = cv2.waitKey(1)

			if key == ord('s') : 
				face_locations = face_recognition.face_locations(rgb_small_frame)
				
				if face_locations != []:
					embeddings = updateEmbeddings(rgb_small_frame, ref_id, embeddings)

					webcam.release()
					cv2.waitKey(1)
					cv2.destroyAllWindows()     
					break

			elif key == ord('q'):
				webcam.release()
				print("Camera off.")
				cv2.destroyAllWindows()
				break

	return embeddings

def addEmbeddingsFromFile(filename, ref_id, embeddings):

	image = cv2.imread(filename)
	small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25) # Resize image for better performance
	rgb_small_image = small_image[:, :, ::-1] # convert the frame from GBR (OpenCV) to RGB 

	face_locations = face_recognition.face_locations(rgb_small_image)

	if face_locations != []:
		embeddings = updateEmbeddings(rgb_small_image, ref_id, embeddings)
	
	return embeddings

def main():
	
	embeddings = readEmbeddingsFromDatabase()

	method = input("Which method would you like to use? Type \"camera\" or \"file\"\n")

	if method == "camera":
		name = input("Enter name: ")
		db, ref_id = updateDatabase(name)
		embeddings = addEmbeddingsFromCamera(ref_id, embeddings)
	
	elif method == "file":
		root = Tk()
		root.withdraw()
		filename = askopenfilename(title='Select Image file')
		
		name = ntpath.basename(filename).replace('-', ' ')[: -4].title()
		
		db, ref_id = updateDatabase(name)
		print(filename, name)
		embeddings = addEmbeddingsFromFile(filename, ref_id, embeddings)

	addEmbeddingsToDatabase(embeddings)



if __name__ == "__main__":
	main()
