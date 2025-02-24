# retrieval/setup_index.py
import faiss
import numpy as np
from embeddings.embedder import embed_text

# Example list of scenario descriptions
texts = [
    "Scenario: Rainy night on an urban street with pedestrians crossing.",
    "Scenario: Sunny day in a suburban area with moderate traffic.",
    "Scenario: Foggy morning on a rural road with low visibility.",
    # Add additional descriptions as needed...
]

# Generate embeddings for each text
embeddings = np.array([embed_text(text) for text in texts]).astype("float32")
embedding_dim = embeddings.shape[1]

# Create and populate the FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)
print(f"Number of vectors in index: {index.ntotal}")

# Optionally, save the index to disk
faiss.write_index(index, "faiss_index.index")
