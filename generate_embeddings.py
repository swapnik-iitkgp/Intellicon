from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text):
    sentences = text.split('\n')
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings