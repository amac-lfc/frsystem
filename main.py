from vgg_face import VGGFace
from algorithms import findCosineSimilarity, findEuclideanDistance
from keras.models import Model
from keras.preprocessing import image
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np
from face_detector import detect_faces, save
import random
import cv2

DATA_DIRECTORY = '/home/arnur/facialdetection/datasets/trainset/'
PLOTS = '/home/arnur/facialdetection/datasets/plots/'
EPSILON = 0.40

def verifyFace(img1, img2):
    haar_face_cascade = cv2.CascadeClassifier('../util/opencv/haarcascade_frontalface_default.xml')

    faces_list = detect_faces(haar_face_cascade, cv2.imread(DATA_DIRECTORY + img1))
    faces_list2 = detect_faces(haar_face_cascade, cv2.imread(DATA_DIRECTORY + img2))
   
    for face in faces_list:
        plot_img1 = save(face, DATA_DIRECTORY, img1)

    for face in faces_list2:
        plot_img2 = save(face, DATA_DIRECTORY, img2)    

    model = VGGFace()
    vgg_face_descriptor = Model(inputs=model.model.layers[0].input, outputs=model.model.layers[-2].output)

    img1_vector = vgg_face_descriptor.predict(model.preprocess_image(DATA_DIRECTORY + str(plot_img1)))[0,:]
    img2_vector = vgg_face_descriptor.predict(model.preprocess_image(DATA_DIRECTORY + str(plot_img2)))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_vector, img2_vector)
    euclidean_distance = findEuclideanDistance(img1_vector, img2_vector)
    
    print("Cosine similarity: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)
    
    if(cosine_similarity < EPSILON):
        print("verified identity match!")
    else:
        print("not verified identity match!")
    
    plot_faces(DATA_DIRECTORY, plot_img1, plot_img2)

def plot_faces(source_folder, image1, image2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img(source_folder + str(image1)))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img(source_folder + str(image2)))
    plt.xticks([]); plt.yticks([])
    plt.savefig(PLOTS + image1[:-4] + "_" + image2[:-4] + '_plot.png')
    plt.show()
    print("-----------------------------------------")

if __name__ == "__main__":
    verifyFace("arnur1.png", "ais1.png")
    verifyFace("steven.jpg", "steven1.jpg")
    verifyFace("sam1.png", "sam2.png")
    verifyFace('ais1.png', 'ais2.png')
    verifyFace("sam1.png", "arnur1.png")
    verifyFace("sam1.png", "arnur3.png")



