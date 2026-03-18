import os
import io
import fitz
import faiss
import numpy as np
import torch
import gradio as gr
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# Initialize Clients and Models
client = Groq(api_key=GROQ_API_KEY)

print("Loading SentenceTransformer model...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Global state
current_chunks = []
current_index = None

def process_new_pdf(file_obj):
    global current_chunks, current_index
    if file_obj is None:
        return "Please upload a file."
    
    try:
        doc = fitz.open(file_obj.name)
        text_chunks = []
        images = []
        
        for page in doc:
            text_chunks.append(page.get_text())
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                images.append(base_image["image"])
                
        def chunk_text(text, size=500):
            return [text[i:i+size] for i in range(0, len(text), size)]
        
        new_chunks = []
        for t in text_chunks:
            new_chunks.extend(chunk_text(t))
            
        if not new_chunks:
            return "No text found in the PDF."
            
        new_embeddings = text_model.encode(new_chunks)
        dim = new_embeddings.shape[1]
        new_index = faiss.IndexFlatL2(dim)
        new_embeddings_np = np.array(new_embeddings).astype('float32')
        new_index.add(new_embeddings_np)
        
        current_chunks = new_chunks
        current_index = new_index
        
        return f"Success! Processed {doc.page_count} pages and created {len(new_chunks)} searchable chunks. Ready for questions."
    except Exception as e:
        return f"Error processing PDF: {e}"

def retrieve_current(query, k=3):
    if current_index is None:
        return []
    q_emb = text_model.encode([query])
    q_emb_np = np.array(q_emb).astype('float32')
    distances, indices = current_index.search(q_emb_np, k)
    return [current_chunks[i] for i in indices[0]]

def generate_answer(context, query):
    prompt = f"""
    Answer the question using the context provided below. 
    If the answer is not in the context, say "I cannot find the answer in the document."

    Context:
    {context}

    Question:
    {query}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq API: {e}"

def chat_with_pdf(query):
    if not current_chunks:
        return "Please upload and process a PDF first."
    
    retrieved_texts = retrieve_current(query)
    context = "\n\n".join(retrieved_texts)
    return generate_answer(context, query)

# Build the UI
with gr.Blocks(title="Multimodal Enterprise Document Assistant") as iface:
    gr.Markdown("# Multimodal Enterprise Document Assistant")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload New PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF")
            upload_status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            query_input = gr.Textbox(label="Ask a question", lines=3)
            submit_btn = gr.Button("Submit")
            answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)
            
    process_btn.click(fn=process_new_pdf, inputs=pdf_input, outputs=upload_status)
    submit_btn.click(fn=chat_with_pdf, inputs=query_input, outputs=answer_output)

if __name__ == "__main__":
    iface.launch()