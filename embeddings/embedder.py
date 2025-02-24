# embeddings/embedder.py
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # Mean pooling over token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()[0]

if __name__ == "__main__":
    sample_text = "Scenario: Rainy night in an urban setting with heavy pedestrian traffic."
    embedding = embed_text(sample_text)
    print("Embedding shape:", embedding.shape)
