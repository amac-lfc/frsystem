import pickle
from .vggface import VGGFace
from .helper import getEmbeddingsList
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import os
def embeddingsPredictor(which="facenet", path=None): 
    
    if which is "vggface":
        model = VGGFace()
        model = Model(model.layers[0].input, model.layers[-2].output)
    elif which is "facenet":
        model = load_model(path)
    else:
        raise AttributeError("invalid attribute. Please use 'vggface' or 'facenet'.")    
    
    return model

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