"""
*************************************************************

************************************************************* 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import ntpath
import cv2

from PIL import ExifTags, Image
from collections import defaultdict
from mtcnn.mtcnn import MTCNN
from vggface import VGGFace
from keras.applications.imagenet_utils import preprocess_input
from algorithms import findCosineScore
from faceAligner import faceAligner

PLOTS = '/Users/newuser/Projects/facialdetection/Identity-Match/datasets/plots'

def detectFaces(filename, required_size=(224, 224)):
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

	# load image from file
    pixels = cv2.imread(filename)
    # convert color channels format from BGR(OpenCV convention) to RGB 
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

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
        plt.imshow(face)
        plt.show()
        # align the face for better precision
        
        landmarks = results[i]["keypoints"]
        face = faceAligner(pixels, landmarks, results[i]['box'])
        
        plt.imshow(face)
        plt.show()
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

def getFaceEmbeddings(filename):
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
    faces_dict = detectFaces(filename)

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
        name = ntpath.basename(filename).replace('-', ' ')[: -4].title()

        # append name to the list of names in our dictionary
        face_embeddings_dict["name"].append(name)

        # append face embedding to the list of embeddings in our dictionary
        face_embeddings_dict["embedding"].append(face_embedding)

        # append bound box to the list of bound boxes in our dictionary
        boundbox = faces_dict[face]["bound_box"]
        face_embeddings_dict["boundbox"].append(boundbox)
        
    return face_embeddings_dict

def isMatch(known_embedding, candidate_embedding, threshold=0.555):
    """
    This function computes cosine distance between two face embeddings

    ## Args:
        known_embedding (array): known face embedding array
        candidate_embedding (array): unknown face embedding array
        threshold (float, optional): identity match threshold. Defaults to 0.555.

    ## Returns:
        tuple: tuple consisting of a boolean and string describing whether the given face embeddings are a match
    """
   
    known_embedding = known_embedding[1:-1].split(", ")
    known_embedding = np.asarray(known_embedding, dtype=np.float)
    
    candidate_embedding = np.asarray(candidate_embedding, dtype=np.float)
    candidate_embedding = np.squeeze(candidate_embedding)
	# calculate distance between embeddings
    score = findCosineScore(known_embedding, candidate_embedding)
    score = round(score, 3)

    if score <= threshold:
        info_string = "face is a Match (" + str(score) + " <= " + str(threshold) + ")"
        result = (True, info_string)
    else:
        info_string = "face is NOT a Match (" + str(score) + " > " + str(threshold) + ")"
        result = (False, info_string)

    return result

def openImage(image):
    """
    This function solves a problem of inverted orientation of an image. Without it, pyplots might show differently

    Args:
        image (string): path to image

    Returns:
        array: processed image for plotting
    """

    try:
        im = Image.open(image)

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        exif=dict(im._getexif().items())

        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        elif exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        elif exif[orientation] == 8:
            im = im.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
    # cases: image don't have getexif
        pass
    
    return im

def plotFaces(image, info_dict):
    """
    This function plots an image with detected names and rectangles around detected faces

    ## Args:
        image (string): path to image
        info_dict (dict): [description]
    """
    im = openImage(image)
    plt.imshow(im)
    ax = plt.gca()
   
    for i in range(len(info_dict["name"])):

        x1 = info_dict["boundbox"][i][0]
        y1 = info_dict["boundbox"][i][1]
        
        x2 = info_dict["boundbox"][i][2]
        y2 = info_dict["boundbox"][i][3]

        name = info_dict["name"][i]

       
        rect = patches.Rectangle((x1,y1),
                                x2 - x1,
                                y2 - y1,
                                linewidth=2,
                                edgecolor='red' if name == 'Unknown' else 'blue',
                                fill = False)

        ax.add_patch(rect)

        plt.text(x1, y1 - 30, name, fontdict={"color" : "white"}, bbox=dict(facecolor='red' if name == 'Unknown' else 'blue', alpha=1))
        
    #plt.savefig(image[:-4] + '_plot.jpg')
    plt.show()

