import faiss
import json
import numpy as np
import ollama
import time
import os
import fitz  # PyMuPDF for PDF text extraction
from sentence_transformers import SentenceTransformer

# Initialize embedding models
embedding_models = {
    "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
}

VECTOR_DIMS = {
    "nomic": 768,
    "minilm": 384,
    "mpnet": 768,
}

def get_embedding(text: str, model_name: str) -> list:
    """Generate embeddings using the specified model."""
    model = embedding_models.get(model_name)
    start_time = time.time()
    
    if model_name == "nomic":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response["embedding"]
    else:
        if model is None:
            raise ValueError(f"Model {model_name} not found!")
        embedding = model.encode(text, normalize_embeddings=True).tolist()
    
    elapsed_time = time.time() - start_time
    return embedding, elapsed_time

def store_embedding(index, file, page, chunk, text, model_name, embeddings_list, metadata_list):
    """Store embeddings into the FAISS index."""
    embedding, elapsed_time = get_embedding(text, model_name)
    embedding = np.array(embedding, dtype=np.float32)
    
    if embedding.shape[0] != index.d:
        raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match FAISS index dimension {index.d}")
    
    embeddings_list.append(embedding)
    metadata_list.append({
        "file": file,
        "page": page,
        "chunk": chunk,
        "text": text,
        "time": elapsed_time
    })
    print(f"Stored embedding for: {text[:30]}...")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file by page."""
    doc = fitz.open(pdf_path)
    text_by_page = [(page_num, page.get_text()) for page_num, page in enumerate(doc)]
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def ingest_documents(index, data_dir, embedding_model):
    """Process and ingest all PDFs in the specified directory into FAISS index."""
    embeddings_list, metadata_list = [], []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    store_embedding(index, file_name, page_num, chunk_index, chunk, embedding_model, embeddings_list, metadata_list)
            
            print(f"✅ Processed: {file_name}")
    
    index.add(np.vstack(embeddings_list))
    return metadata_list

def search_embeddings(index, query, metadata_list, embedding_model="minilm", top_k=5):
    """Retrieve top-k similar chunks using FAISS."""
    query_embedding, elapsed_time = get_embedding(query, embedding_model)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    distances, indices = index.search(query_embedding, top_k)
    top_results = []
    
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata_list):
            top_results.append({
                "file": metadata_list[idx]["file"],
                "page": metadata_list[idx]["page"],
                "chunk": metadata_list[idx]["chunk"],
                "text": metadata_list[idx]["text"],
                "similarity": distances[0][i],
            })
    
    print(f"🔍 Search completed in {elapsed_time:.4f} sec using {embedding_model}")
    return top_results[:top_k]

def generate_rag_response(query, context_results, ollama_model="mistral"):
    """Generate response using retrieved context."""
    context_str = "\n".join([f"From {r['file']} (Page {r['page']}): {r['text']}" for r in context_results])
    prompt = f"""
    Use the following context to answer the query as accurately as possible:
    Context:
    {context_str}
    Query: {query}
    Answer:"""
    
    response = ollama.chat(model=f"{ollama_model}:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    while True:
        embedding_model = input("Enter embedding model (nomic/minilm/mpnet): ").strip().lower()
        if embedding_model in VECTOR_DIMS:
            break
        print("Invalid choice. Try again.")
    
    VECTOR_DIM = VECTOR_DIMS[embedding_model]
    index = faiss.IndexFlatL2(VECTOR_DIM)
    metadata_list = []
    
    file_path = "/Users/tristanco/Desktop/DS4300_Practical_2/Data/"
    print("Ingesting PDFs...")
    metadata_list = ingest_documents(index, file_path, embedding_model)
    print("Document ingestion complete.")
    
    while True:
        ollama_model = input("Choose Ollama model (mistral/llama3.2): ").strip().lower()
        if ollama_model not in ["mistral", "llama3.2"]:
            print("Invalid choice. Using 'mistral' by default.")
            ollama_model = "mistral"
        
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        context_results = search_embeddings(index, query, metadata_list, embedding_model)
        response = generate_rag_response(query, context_results, ollama_model)
        print("\n--- Response ---")
        print(response)
