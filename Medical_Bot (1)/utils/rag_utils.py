# rag_utils.py
import faiss # type: ignore
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer # type: ignore

# Load FAISS and documents
index = faiss.read_index('vectorstore/faiss_index.bin')
with open('vectorstore/documents.pkl', 'rb') as f:
    documents = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_context(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [documents[i] for i in indices[0]]
    return "\n".join(results)