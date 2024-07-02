import numpy as np

def search_index(query, index, model, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    
    return I[0] 