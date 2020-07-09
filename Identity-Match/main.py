import numpy as np
import pandas as pd
import cv2
import csv
import os
# Run Tensorflow on CPU
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from faceDetector import getFaceEmbeddings, isMatch, plotFaces

KNOWN_FACES = '/Users/newuser/Projects/facialdetection/Identity-Match/datasets/known_faces'

def extractKnownFacesToCSV(folder):
    """
    This function extracts faces from images in a given folder and saves them as known_faces.csv

    ## Args:
        folder (string): path to the folder containing images of known identities
    """
    fields = ["Name", "Embedding", "Bound Box"]

    with open("known_faces.csv", "w") as csvfile:

        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        
        # writing the fields  
        csvwriter.writerow(fields)
        
        for f in os.listdir(folder):
            # ignoring hidden files
            if f[0] == ".":
                continue

            embedding_dict = getFaceEmbeddings(os.path.join(folder,f))

            embedding = list(embedding_dict["embedding"][0][0])

            csvwriter.writerow([ embedding_dict["name"][0], embedding, embedding_dict["boundbox"][0][0] ])

def detectIdentities(filename, showFigure=False, printInfo=False):
    """
    This function detects identites in an image, if identity is unknown it returns "Unknown"

    ## Args:
        filename (string): path to the image 
        showFigure (bool): whether to show the resulting figure
        printInfo (bool): whether to print face detection info

    ## Returns
        list of detected names
    """
    
    try:
        df = pd.read_csv("known_faces.csv")
    except FileNotFoundError:
        print("known_faces.csv not found.")

    known_embeddings = df.to_dict('r')

    unknown_embeddings = getFaceEmbeddings(filename)

    detected_names_dict = {"name" : [], "boundbox" : []}

    for i in range(len(unknown_embeddings["name"])):

        for j in range(len(known_embeddings)):

            (is_match, info_string) = isMatch(known_embeddings[j]["Embedding"], unknown_embeddings["embedding"][i])
            
            if printInfo:
                print(info_string)

            if is_match:
                detected_name = known_embeddings[j]["Name"]

                if detected_name in detected_names_dict["name"]:
                    continue

                break   
        else:
            detected_name = "Unknown"
            
        boundbox = unknown_embeddings["boundbox"][i][0]
        detected_names_dict["boundbox"].append(boundbox)
        detected_names_dict["name"].append(detected_name)

    if showFigure:
        plotFaces(filename, detected_names_dict)
    
    return detected_names_dict["name"]   


""" 
Run extractKnownFacesToCSV to create a csv file of Known Faces, comment the call to detectIdentities below before running.
"""
#extractKnownFacesToCSV(KNOWN_FACES)

print(detectIdentities("/Users/newuser/Projects/facialdetection/Identity-Match/datasets/unknown_faces/unknown5.jpg", showFigure=True, printInfo=True))




