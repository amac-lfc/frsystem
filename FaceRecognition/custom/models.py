from tensorflow.keras import Model
from vggface import VGGFace
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

FACENET_MODEL = "/Users/newuser/Projects/facialdetection/FaceRecognition/custom/util/facenet_keras.h5"

def embeddingsPredictor(which="facenet"): 
    
    if which is "vggface":
        model = VGGFace()
        model = Model(model.layers[0].input, model.layers[-2].output)
    elif which is "facenet":
        model = load_model(FACENET_MODEL)
    else:
        raise AttributeError("invalid attribute. Please use 'vggface' or 'facenet'.")    
    
    return model

