# 🚗 Synthetic Image Generation for Autonomous Vehicle Scenarios

This project provides an end-to-end solution for generating **synthetic images of autonomous vehicle (AV) scenarios** using a **Retrieval-Augmented Generation (RAG)** pipeline. It leverages scenario metadata, LangChain for retrieval, FAISS for similarity search, and **Stable Diffusion** for high-quality image generation.

The system allows you to input a text description of a driving scenario, retrieves related prompts from a dataset of scenarios, and generates high-resolution synthetic images based on the input and retrieved information.

---

## 📂 Project Structure

``` bash
synthetic-data-av/
├── data/
│   └── scenario_prompts.csv            # Scenario prompts used for retrieval
├── generation/
│   ├── image_generation.py             # Image generation using Stable Diffusion
│   └── rag_pipeline.py                 # Retrieval-Augmented Generation pipeline
├── retrieval/
│   └── langchain_integration.py        # FAISS-based vector store using LangChain
├── streamlit_app.py                    # Interactive UI for image generation
├── environment.yml                     # Conda environment dependencies
├── requirements.txt                    # pip dependencies
└── README.md                           # Project documentation (this file)
```

---

## 🔧 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <repository_url>
cd synthetic-data-av
```

### 2️⃣ Create a Conda Environment (Recommended)

Create and activate a Conda environment for dependency management:

```bash
conda env create -f environment.yml
conda activate synthetic
```


## 🚀 How to Run the Project

### 1️⃣ Prepare Scenario Metadata

Create a CSV file (`data/scenario_prompts.csv`) with sample AV scenario prompts such as:

```
id,weather,location,time,description
1,rain,urban,night,"Scenario: Rainy night in an urban area with busy pedestrian crossings and low visibility."
2,sunny,suburban,day,"Scenario: Sunny day in a suburban area with moderate traffic and clear skies."
3,fog,rural,morning,"Scenario: Foggy morning on a rural road with low visibility and sparse traffic."
```

---

### 2️⃣ Test Image Generation (Standalone)

You can test Stable Diffusion image generation by running:

```bash
python generation/image_generation.py
```

This will generate an image based on a predefined prompt and save it as `generated_sd_image.png`.

---

### 3️⃣ Run the RAG Pipeline

To retrieve similar prompts and generate an augmented scenario image:

```bash
python generation/rag_pipeline.py
```

The resulting image will be saved as `rag_generated_image.png`.

---

### 4️⃣ Launch the Streamlit App (Interactive Mode)

To generate images interactively:

```bash
streamlit run streamlit_app.py
```

Open the local URL (usually `http://localhost:8501`) and input a description (e.g., `"An autonomous vehicle driving on a rainy night in a busy urban area."`). The app will retrieve similar prompts from the dataset and generate an image.

---

## ⚡ Performance Optimization

- **GPU Support:**  
  Remove or comment out `pipe.to("cpu")` in `image_generation.py` if you have a GPU for faster image generation.

- **Lower Resolution for Faster Generation:**  
  Modify the `height` and `width` parameters in `generate_image()` to `512` for faster inference:
  ```python
  height=512, width=512
  ```

- **Reduce Inference Steps:**  
  Decrease `num_inference_steps` for faster, though lower-quality, results.

---

## 🛠️ Troubleshooting

- **`Torch not compiled with CUDA enabled`:**  
  Make sure your PyTorch installation has GPU support. Reinstall PyTorch with CUDA:
  ```bash
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
  ```

- **`sentencepiece` Build Errors:**  
  Install `cmake` before reinstalling `sentencepiece`:
  ```bash
  brew install cmake  # macOS
  sudo apt-get install cmake pkg-config  # Ubuntu
  ```

- **Streamlit Deprecation Warning:**  
  Replace:
  ```python
  st.image(image, use_column_width=True)
  ```
  with:
  ```python
  st.image(image, use_container_width=True)
  ```

---

## 🔮 Future Work (Checklist)

- [ ] Integrate real-world datasets like KITTI and nuScenes for more realistic scenario metadata.
- [ ] Optimize the RAG pipeline for faster retrieval performance.
- [ ] Add support for more advanced image generation models (e.g., Stable Diffusion XL).
- [ ] Implement GPU memory optimization for large-scale deployments.
- [ ] Create a Docker container for easier deployment across platforms.
- [ ] Integrate logging and monitoring for generated outputs.
- [ ] Develop a benchmarking suite to measure performance across various hardware setups.
- [ ] Enable dynamic configuration of hyperparameters through the Streamlit app.
- [ ] Implement batch image generation for efficiency in larger datasets.
- [ ] Add automated testing for different components (retrieval, generation, UI).

---

## 📄 License

This project is licensed under the CC0 1.0 Universal License.

---
