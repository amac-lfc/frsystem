# Facial Detection
## About
This project involves the study of convolutional neural networks using Keras library. It uses one of existing CNN architectures to detect faces on a given image and classify them using Logistic Regression first, and then using a Neural Network. After the initial step of classifying identities from an image, the project then proceeds to implementing a "Celebrity Doppelganger" finder, which uses imdb dataset of celebrities. Since, imdb dataset has 'age' and 'gender' labels, the project will further add age and gender prediciton functionality.
## Resources
[Andrew Ng Tutorials](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=1 "Andrew Ng")
to be added...
## Data
[Labeled Faces in The Wild](http://vis-www.cs.umass.edu/lfw/ "original dataset")
[imdb Celebrity Images](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ "imdb dataset")
## Algorithm of implementation
* Step 1: Explore Logistic regression in Keras. 
* Step 2: Choose the appropriate CNN architecture and examine it. 
* Step 3: Prepare custom dataset (can be done after testing on LFW dataset). 
* Step 4: Take care of cropping and face alignment using dlib and OpenCV libraries. 
* Step 5: Feed the aligned and scaled images into the pre-trained network and calculate output vectors. 
* Step 6: Identify threshold for classifying identities. At a given threshold, all possible output vector pairs are classified as either same identity or different identity and compared to the ground truth. 
* Step 7: Given an estimate of the distance threshold, face recognition can be done by calculating the distances between an output vector and all output vectors in the database. This will be done using the classifier from step 1 and a NN. 
* Step 8: Train the chosen CNN on celebrity image data and calculate their output vectors. As we know, output vector is the output we get from a pretrained NN which detects a face on an image. 
* Step 9: Compare the output vectors of celebrities to the output vector of the custom image data (the one we want to find a look alike for) by finding Cosine Similarity. 
