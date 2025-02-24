# retrieval/langchain_integration.py
import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Define the path to your CSV file
DATA_FILE = os.path.join("data", "scenario_prompts.csv")

# Load the CSV file using pandas
df = pd.read_csv(DATA_FILE)

# Extract the scenario descriptions (the "description" column)
texts = df["description"].tolist()

# Initialize the embeddings model using LangChain's wrapper
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Build a FAISS vector store from the texts
vectorstore = FAISS.from_texts(texts, embedding_model)

if __name__ == "__main__":
    # Example query to test retrieval
    query = "Generate a scenario for a rainy night in an urban area."
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    results = retriever.get_relevant_documents(query)
    for doc in results:
        print(doc.page_content)
