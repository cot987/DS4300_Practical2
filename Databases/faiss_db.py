import faiss
import json
import numpy as np
import ollama
import time
import os
import fitz  # PyMuPDF for PDF text extraction
from sentence_transformers import SentenceTransformer

# FAISS index settings
VECTOR_DIM = 768  # Update this with the appropriate embedding vector size
INDEX_FILE_PATH = "faiss_index.index"

# Initialize embedding models
embedding_models = {
    "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    # "instructorxl": SentenceTransformer("hkunlp/instructor-xl")
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
    embedding = np.array(embedding).astype('float32')

    # Add the embedding and metadata to the lists
    embeddings_list.append(embedding)
    metadata_list.append({
        "file": file,
        "page": page,
        "chunk": chunk,
        "text": text,
        "time": elapsed_time
    })

    print(f"Stored embedding for: {text}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file by page."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_documents(index, data_dir, embedding_model):
    """Process and ingest all PDFs in the specified directory into FAISS index."""
    embeddings_list = []
    metadata_list = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    store_embedding(index, file_name, page_num, chunk_index, chunk, embedding_model, embeddings_list, metadata_list)

            print(f"✅ Processed: {file_name}")

    # Convert list to numpy array and store in FAISS index
    embeddings_array = np.array(embeddings_list).astype('float32')
    index.add(embeddings_array)

    return metadata_list

def search_embeddings(index, query, embedding_model="minilm", top_k=5):
    """Retrieve top-k similar chunks using FAISS."""
    query_embedding, elapsed_time = get_embedding(query, embedding_model)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    # Search for the top_k most similar embeddings in the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    if len(distances[0]) == 0:
        print("No relevant results found.")
        return []

    top_results = []
    for i, idx in enumerate(indices[0]):
        top_results.append({
            "file": metadata_list[idx].get("file", "Unknown"),
            "page": metadata_list[idx].get("page", "Unknown"),
            "chunk": metadata_list[idx].get("chunk", "Unknown"),
            "text": metadata_list[idx].get("text", "No text available"),
            "similarity": distances[0][i]
        })

    print(f"🔍 Search completed in {elapsed_time:.4f} sec using {embedding_model}")

    for result in top_results:
        print(
            f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['text']}"
        )

    return top_results[:top_k]

def generate_rag_response(query, context_results, ollama_model="mistral"):
    """Generate response using retrieved context."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (Page {result.get('page', 'Unknown page')}, Chunk {result.get('chunk', 'Unknown chunk')}):\n{result.get('text', '')}\n"
            f"Similarity: {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.
    
Context:
{context_str}

Query: {query}

Answer:"""
    
    response = ollama.chat(
        model=f"{ollama_model}:latest", messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]

if __name__ == "__main__":
    # Initialize FAISS index
    index = faiss.IndexFlatL2(VECTOR_DIM)

    metadata_list = []

    while True:
        embedding_model = input("Enter the embedding model to use for ingestion and querying (nomic/minilm/mpnet): ").strip().lower()
        if embedding_model in ['nomic', 'minilm', 'mpnet']:
            break  # Exit the loop when a valid model is entered
        print("Invalid model. Please choose from: nomic, minilm, mpnet.")

    file_path = "/Users/tristanco/Desktop/DS4300_Practical_2/Data/"
    print("Ingesting PDFs")
    metadata_list = ingest_documents(index, file_path, embedding_model)
    print("Document Ingestion Complete")

    # Save the FAISS index to disk
    faiss.write_index(index, INDEX_FILE_PATH)
    print(f"FAISS index saved to {INDEX_FILE_PATH}")

    while True:
        print(f"\n\n\n----- Querying from FAISS index that uses: {embedding_model}")
        ollama_model_choice = input("Choose Ollama model to provide context to (mistral/llama3.2): ").strip().lower()
        if ollama_model_choice not in ["mistral", "llama3.2"]:
            print("Invalid Ollama model choice. Using 'mistral' by default.")
            ollama_model_choice = "mistral"

        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        context_results = search_embeddings(index, query, embedding_model)

        response = generate_rag_response(query, context_results, ollama_model_choice)
        
        print("\n--- Response ---")
        print(response)
