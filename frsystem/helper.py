def getEmbeddingsList(embeddings_dict):
    """
    ### Decription: 
        Given dictionary of embeddings return two lists 
        one containing ids of known people and the other containing face embeddings
        
    ```python
        dict = {
            1 : [ [embedding1], [embedding2], [embeddingN] ]
            2 : [ [embedding1], [embedding2], [embeddingN] ]
            3 : [ [embedding1], [embedding2], [embeddingN] ]
        }
        
        returns 
        embeddings_list = [ [embedding1], [embedding2], [embedding3], [embeddingN] ]
        id_list = [1, 1, 1, 1, 1, 2, 2, 2, ...]
    ```   
    ### Args
        embeddings (dict, optional): dictionary where keys are ids of known people and values are their embeddings. Defaults to None.

    ### Returns:  
        two lists with all ids and embeddings.
    """
    embeddings = embeddings_dict
    
    embeddings_list = []  # known face embeddings
    id_list = []	   # unique ids of known face embeddings


    for ref_id , embed_list in embeddings.items():
        if len(embed_list) > 1:
            for e in embed_list:
                embeddings_list.append(e)
                id_list.append(ref_id)
        else:
            embeddings_list.append(embed_list[0])
            id_list.append(ref_id)
    
    return embeddings_list, id_list