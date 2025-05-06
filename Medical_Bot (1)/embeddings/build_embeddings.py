# build_embeddings.py
import os
import faiss  # type: ignore
import pickle
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np

# 1. Load your medical documents (for now, we use simple list)
documents = ["Medical.pdf.pdf"]

# 2. Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Create embeddings
embeddings = model.encode(documents)

# 4. Save documents
with open('vectorstore/documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

# 5. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 6. Save FAISS index
faiss.write_index(index,'vectorstore/faiss_index.bin')
print("Embeddings and FAISS index saved successfully!")
