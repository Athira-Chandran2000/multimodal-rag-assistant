```markdown
```
# Multimodal Enterprise Document Assistant (RAG)

I architected and deployed a Multimodal Retrieval-Augmented Generation (RAG) system that allows users to interactively query complex enterprise PDFs containing both text and images. Built primarily with Python, the backend leverages a dual-embedding strategy using HuggingFace models (Sentence-Transformers and CLIP), storing the high-dimensional vectors in a local FAISS database for highly efficient semantic search. For the generative component, I integrated the Groq API (running LLaMA 3.1) to ensure ultra-low latency inference. Users interact with the pipeline through an intuitive Gradio web interface, where they can dynamically upload documents and receive context-grounded, natural language answers.

## System Methodology & Architecture

The system follows a local-to-cloud RAG pipeline tailored for multimodal documents:

1. **Document Ingestion & Parsing (`PyMuPDF`)**
   * Ingests enterprise PDF documents.
   * Extracts raw text page-by-page.
   * Extracts embedded images and converts them into byte streams.
2. **Chunking Strategy**
   * Text is divided into fixed-size overlapping chunks to preserve local context without exceeding embedding context windows.
3. **Multimodal Embeddings (`Sentence-Transformers` & `OpenAI CLIP`)**
   * **Text:** Encodes text chunks into dense 384-dimensional vectors using `all-MiniLM-L6-v2`.
   * **Images:** Processes extracted images through `clip-vit-base-patch32` to generate semantic image embeddings.
4. **Vector Storage (`FAISS`)**
   * Embeddings are stored in a local, high-speed FAISS (`IndexFlatL2`) database for sub-millisecond similarity search.
5. **Retrieval & Context Generation**
   * User queries are encoded using the identical text embedding model.
   * Performs an L2 distance search against the FAISS index to retrieve the top-k most relevant chunks.
6. **Generative Answering (`Groq API` / `LLaMA 3.1 8B`)**
   * The retrieved context is formatted into a strict prompt.
   * The LLaMA 3.1 model via Groq processes the context and query to generate a synthesized, fact-based response, with explicit instructions to fallback if the answer is not present in the document.
7. **Interactive UI (`Gradio`)**
   * Provides a web-based interface for dynamic document uploading, background embedding generation, and real-time inference.

## Tech Stack
* **Language:** Python 3.10+
* **LLM:** Groq API (LLaMA 3.1 8B Instant)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Models:** HuggingFace `sentence-transformers`, OpenAI `CLIP`
* **Document Parsing:** PyMuPDF (`fitz`), Pillow
* **Frontend UI:** Gradio
```
## Local Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd multimodal-rag
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows (Git Bash): source venv/Scripts/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 5. Run the Application
```bash
python app.py
```
Open the local URL provided in the terminal (typically `http://127.0.0.1:7860`) in your web browser.

## Usage
1. Open the web interface.
2. Upload a PDF document using the file uploader.
3. Click **Process PDF** and wait for the status to show "Success!".
4. Type your question in the text box and click **Submit** to retrieve context-aware answers.
```
