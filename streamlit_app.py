import streamlit as st
from generation.rag_pipeline import generate_rag_image
from io import BytesIO

st.set_page_config(page_title="AV Synthetic Image Generator", layout="wide")
st.title("Synthetic Image Generation for Autonomous Vehicle Scenarios")

st.markdown("""
This tool uses a RAG pipeline with Stable Diffusion model to generate synthetic images for AV scenarios based on your prompt.
""")

# Input field for the prompt
query = st.text_input("Enter a description of the driving scenario:",
                        placeholder="e.g., A futuristic autonomous vehicle on a neon-lit urban street at night.")

if st.button("Generate Image"):
    if not query:
        st.error("Please enter a prompt describing the scenario.")
    else:
        with st.spinner("Generating image..."):
            image = generate_rag_image(query, k=2)
        st.success("Image Generated!")
        buf = BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.image(byte_im, caption="Generated Synthetic Image", use_container_width=True)
