import re
import os
import ntpath
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from frs import FaceRecognitionSystem

DB = "/Users/newuser/Projects/facialdetection/FaceRecognition/custom/data/db.pkl"
EMBEDDINGS = "/Users/newuser/Projects/facialdetection/FaceRecognition/custom/data/embeddings.pkl"

def addFaceToDatabase():
	
    fr = FaceRecognitionSystem(160, DB, EMBEDDINGS) 

    method = input("Which method would you like to use? Type \"camera\" or \"file\"\n")

    while method not in ("camera", "file"):
        method = input("Which method would you like to use? Type \"camera\" or \"file\"\n")
        
    if method == "camera":
	    name = input("Enter name: ")
	    fr.addEmbeddingsFromCamera(name)

    else:
        regex = re.compile('[0-9_\.\/_]+.jpg')
        root = Tk()
        root.withdraw()
        filename = askopenfilename(title='Select Image file')

        correct_filename = input("Does your filename follow the following 'name-lastname.jpg' naming convention? (y/n) ")
        while correct_filename not in ("y", "n", "Y", "N"):
            correct_filename = input("Does your filename follow the following 'name-lastname.jpg' naming convention? (y/n) ")
        
        if correct_filename in ("y", "Y"): 
            #First parameter is the replacement, second parameter is your input string
            path_name = ntpath.basename(filename)
            name = regex.sub('', "{}".format(filename)).replace("_", " ").title()
        elif correct_filename in ("n", "Y"):
            name = input("Please enter name as the following -> Name Lastname: ")
        
        fr.addEmbeddingsFromFile(filename, name)

def addFacesUsingLoop(base):
    
    fr = FaceRecognitionSystem(160, DB, EMBEDDINGS)
    
    for folder in os.listdir(base):
        if folder[0] == ".":
            continue

        path = os.path.join(base, folder)
        for image in os.listdir(path):
            if image[0] == ".":
                continue
            fr.addEmbeddingsFromFile(os.path.join(path, image), folder)

if __name__ == "__main__":
    action = int(input("add faces to database manually or through folder loop? type '1' or '2'"))

    if action == 1:
        addFaceToDatabase() 
    elif action == 2:
        addFacesUsingLoop("/Users/newuser/Projects/facialdetection/FaceRecognition/custom/jpg")  
    else:
        print("Invalid input. Try again.")