from keras.models import Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageFile
import numpy as np

class VGGFace:

  def __init__(self):
    self.model = Sequential()
    self.model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    self.model.add(Convolution2D(64, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(64, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D((2,2), strides=(2,2)))

    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(128, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(128, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D((2,2), strides=(2,2)))

    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(256, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(256, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(256, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D((2,2), strides=(2,2)))

    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(512, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(512, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(512, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D((2,2), strides=(2,2)))

    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(512, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(512, (3, 3), activation='relu'))
    self.model.add(ZeroPadding2D((1,1)))
    self.model.add(Convolution2D(512, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D((2,2), strides=(2,2)))

    self.model.add(Convolution2D(4096, (7, 7), activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Convolution2D(4096, (1, 1), activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Convolution2D(2622, (1, 1)))
    self.model.add(Flatten())
    self.model.add(Activation('softmax'))

  def setWeights(self, path):
    self.model.load_weights(path)
  
  def preprocess_image(self, path):
    self.img = load_img(path, target_size=(224, 224))
    self.img = img_to_array(self.img)
    self.img = np.expand_dims(self.img, axis=0)
    self.img = preprocess_input(self.img)
    return self.img