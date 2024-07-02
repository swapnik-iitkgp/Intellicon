import faiss
import numpy as np

def store_embeddings(embeddings, index, filename):
    index.add(np.array(embeddings))
    faiss.write_index(index, filename)