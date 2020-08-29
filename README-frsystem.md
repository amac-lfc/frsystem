# frsystem v1.1.0

## FaceRecognitionSystem class

Initializes a **FaceRecognitionSystem** class object.  

1. Calls MTCNN face detection model object.
2. Calls FaceNet or VGGFace model to extract embeddings (features) from face images.

 3.  Creates a connection to the database of known faces by calling the "Database" class object. 

- self.db is the dictionary of known faces with id : name key-value pairs.
- self.embeddings is the dictionary of known faces with id : embeddings key-value pairs.

 4. Loads face classifier that was trained on the database of known faces. 

```python
frsystem.frs.FaceRecognitionSystem(self,
				   embedding_model=None,
				   weights=None,
				   face_classifier=None,
				   **kwargs)
```

### Arguments
Name | Description 
---------- | ---------- |
embedding_model	| Options: <br>  1. **None**. If you want to use only face location and facial features detection functionality.<br> 2. **facenet**. Use FaceNet as the feature extractor model. Input size for FaceNet is 160x160x3 <br> 3. **vggface**. Use VGG-Face as the feature extractor model. Input size for VGG-Face is 224x224x3
weights	| File path to the weights for the chosen embedding model. Defaults to None
face_classifier	| File path to pre-trained face classifier. Face classifier 
**kwargs | Two keyword arguments that are passed to the Database class. **db_file** and **embeddings_file** 

More extended docs coming soon.

See https://github.com/amac-lfc/frsystem/tree/master/frsystem **frs**.**py** file for more information.