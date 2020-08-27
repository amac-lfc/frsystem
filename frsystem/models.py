import pickle
import numpy as np
from .vggface import VGGFace
from .helper import getEmbeddingsList
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression



def embeddingsPredictor(which=None, path=None): 
    
    if which is "vggface":
        face_size = 224
        model = VGGFace(path)
        model = Model(model.layers[0].input, model.layers[-2].output)
    elif which is "facenet":
        face_size = 160
        model = load_model(path)
    else:
        raise AttributeError("invalid attribute. Please use 'vggface' or 'facenet'.")    
    
    return model, face_size

def faceClassifier(embeddings_dict=None, path=None):
        
    """
    ### Description 
        Loads face classifier if serialized model file exists, 
        else trains a face classifier and returns it.

    ### Returns:
        clf: sklearn model object
    """
            
    try:
        with open(path, 'rb') as f:
            clf = pickle.load(f)
    except:
        if embeddings_dict is not None:  
            X, y = getEmbeddingsList(embeddings_dict)
            X = np.array(X)
            print("Training face classifier...")
            clf = LogisticRegression().fit(X, y)  
            with open(path, 'wb') as f:
                pickle.dump(clf, f)
        else:
            raise AttributeError("Invalid embeddings_dict argument.")
    
    return clf