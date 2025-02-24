from retrieval.langchain_integration import vectorstore
from generation.image_generation import generate_image

def generate_rag_image(query: str, k: int = 2):
    # Retrieve similar prompts from the vector store
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Augment the query with retrieved prompts to form a comprehensive prompt
    augmented_prompt = query
    if retrieved_docs:
        retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])
        augmented_prompt += " " + retrieved_text
    
    print("Augmented Prompt:", augmented_prompt)
    
    # Generate an image based on the augmented prompt
    return generate_image(augmented_prompt)

if __name__ == "__main__":
    query = "A futuristic autonomous vehicle driving in a neon-lit cityscape"
    image = generate_rag_image(query, k=2)
    image.save("rag_generated_image.png")
    print("Image saved as rag_generated_image.png")
