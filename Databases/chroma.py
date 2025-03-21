# import chromadb
# import json
# import numpy as np
# import ollama
# import time
# import os
# import fitz  # PyMuPDF for PDF text extraction
# from sentence_transformers import SentenceTransformer

# # Initialize Chroma client and collection
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="ds4300_notes")

# VECTOR_DIM = 768  # Ensure this matches your embedding model

# # def clear_chroma_store():
# #     print("Clearing existing ChromaDb  store...")
# #     chroma_client.clear()
# #     print("ChromaDb store cleared.")

# # Initialize embedding models
# embedding_models = {
#     "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
#     "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
#     "instructorxl": SentenceTransformer("hkunlp/instructor-xl")
# }

# def get_embedding(text: str, model_name: str) -> list:
#     """Generate embeddings using the specified model."""
#     model = embedding_models.get(model_name)
#     if model is None:
#         raise ValueError(f"Model {model_name} not found!")

#     start_time = time.time()
#     embedding = model.encode(text, normalize_embeddings=True).tolist()
#     elapsed_time = time.time() - start_time

#     return embedding, elapsed_time

# def store_embedding(file, page, chunk, text):
#     """Store embeddings for multiple models."""
#     embeddings = {}
#     times = {}

#     for model_name in embedding_models.keys():
#         embedding, elapsed_time = get_embedding(text, model_name)
#         embeddings[model_name] = embedding
#         times[model_name] = elapsed_time

#     doc_id = f"{file}_page_{page}_chunk_{chunk}"
    
#     # Add embeddings only to the embeddings field, while metadata will store scalar values.
#     collection.add(
#         ids=[doc_id],
#         embeddings=[embeddings["minilm"]],  # Default embedding for indexing
#         metadatas=[{
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#             "text": text,
#             "time_minilm": times["minilm"],
#             "time_mpnet": times["mpnet"],
#             "time_instructorxl": times["instructorxl"]
#         }]
#     )

#     print(f"Stored embedding for: {chunk}")

# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file by page."""
#     doc = fitz.open(pdf_path)
#     text_by_page = []
#     for page_num, page in enumerate(doc):
#         text_by_page.append((page_num, page.get_text()))
#     return text_by_page

# def split_text_into_chunks(text, chunk_size=300, overlap=50):
#     """Split text into overlapping chunks."""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i : i + chunk_size])
#         chunks.append(chunk)
#     return chunks

# def ingest_documents(data_dir):
#     """Process and ingest all PDFs in the specified directory."""
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".pdf"):
#             pdf_path = os.path.join(data_dir, file_name)
#             text_by_page = extract_text_from_pdf(pdf_path)

#             for page_num, text in text_by_page:
#                 chunks = split_text_into_chunks(text)
#                 for chunk_index, chunk in enumerate(chunks):
#                     store_embedding(file_name, page_num, chunk_index, chunk)
                    
#             print(f"✅ Processed: {file_name}")

# def search_embeddings(query, model_name="minilm", top_k=5):
#     """Retrieve top-k similar chunks using the selected embedding model."""
#     query_embedding, elapsed_time = get_embedding(query, model_name)

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )

#     if not results["metadatas"] or not results["distances"]:
#         print("No relevant results found.")
#         return []

#     top_results = []
#     for metadata_list, distance_list in zip(results["metadatas"], results["distances"]):
#         for metadata, distance in zip(metadata_list, distance_list):
#             top_results.append({
#                 "file": metadata.get("file", "Unknown"),
#                 "page": metadata.get("page", "Unknown"),
#                 "chunk": metadata.get("chunk", "Unknown"),
#                 "text": metadata.get("text", "No text available"),
#                 "similarity": distance
#             })

#     print(f"🔍 Search completed in {elapsed_time:.4f} sec using {model_name}")

#     return top_results[:top_k]

# def generate_rag_response(query, context_results, ollama_model="mistral"):
#     """Generate response using retrieved context."""
#     context_str = "\n".join(
#         [
#             f"From {result.get('file', 'Unknown file')} (Page {result.get('page', 'Unknown page')}, Chunk {result.get('chunk', 'Unknown chunk')}):\n{result.get('text', '')}\n"
#             f"Similarity: {float(result.get('similarity', 0)):.2f}"
#             for result in context_results
#         ]
#     )

#     prompt = f"""You are a helpful AI assistant. 
#     Use the following context to answer the query as accurately as possible. If the context is 
#     not relevant to the query, say 'I don't know'.
    
# Context:
# {context_str}

# Query: {query}

# Answer:"""
    
#     response = ollama.chat(
#         model=ollama_model, messages=[{"role": "user", "content": prompt}]
#     )
    
#     return response["message"]["content"]

# def interactive_search():
#     """Interactive CLI to search documents and get AI-generated responses."""
#     print("\n🔍 **Chroma RAG Search Interface**")
#     print("Type 'exit' to quit\n")
    
#     while True:
#         query = input("\nEnter your search query: ")
#         if query.lower() == "exit":
#             break
        
#         model_choice = input("Choose embedding model (minilm/mpnet/instructorxl): ").strip().lower()
#         if model_choice not in embedding_models:
#             print("Invalid model choice. Using 'minilm' by default.")
#             model_choice = "minilm"

#         context_results = search_embeddings(query, model_name=model_choice)

#         ollama_model_choice = input("Choose Ollama model (mistral/llama3.2): ").strip().lower()
#         if ollama_model_choice not in ["mistral", "llama3.2"]:
#             print("Invalid Ollama model choice. Using 'mistral' by default.")
#             ollama_model_choice = "mistral"

#         response = generate_rag_response(query, context_results, ollama_model_choice)
        
#         print("\n--- Response ---")
#         print(response)

# if __name__ == "__main__":
#     # Ingest documents first
#     pdf_path = "/Users/CalvinLii/Documents/ds4300/RagIngestAndSearch/data/"
#     print("Ingesting PDFs")

#     # clear_chroma_store() 
#     ingest_documents(pdf_path)
#     print("Document Ingestion Complete")

#     # Start the interactive search
#     interactive_search()



# --------------- Calvin Errors 3/17 have to fix the issues with only selecting on model for both ingestion and the querying 

# import chromadb
# import json
# import numpy as np
# import ollama
# import time
# import os
# import fitz  # PyMuPDF for PDF text extraction
# from sentence_transformers import SentenceTransformer

# # Initialize Chroma client and collection
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="ds4300_notes")

# VECTOR_DIM = 768  # Ensure this matches your embedding model

# # Initialize embedding models
# embedding_models = {
#     "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
#     "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
#     "instructorxl": SentenceTransformer("hkunlp/instructor-xl")
# }

# def get_embedding(text: str, model_name: str) -> list:
#     """Generate embeddings using the specified model."""
#     model = embedding_models.get(model_name)
#     if model is None:
#         raise ValueError(f"Model {model_name} not found!")

#     start_time = time.time()
#     embedding = model.encode(text, normalize_embeddings=True).tolist()
#     elapsed_time = time.time() - start_time

#     return embedding, elapsed_time

# def store_embedding(file, page, chunk, text):
#     """Store embeddings for multiple models."""
#     embeddings = {}
#     times = {}

#     for model_name in embedding_models.keys():
#         embedding, elapsed_time = get_embedding(text, model_name)
#         embeddings[model_name] = embedding
#         times[model_name] = elapsed_time

#     doc_id = f"{file}_page_{page}_chunk_{chunk}"
    
#     # Add embeddings only to the embeddings field, while metadata will store scalar values.
#     collection.add(
#         ids=[doc_id],
#         embeddings=[embeddings["minilm"]],  # Default embedding for indexing
#         metadatas=[{
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#             "text": text,
#             "time_minilm": times["minilm"],
#             "time_mpnet": times["mpnet"],
#             "time_instructorxl": times["instructorxl"]
#         }]
#     )

#     print(f"Stored embedding for: {text}")



# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file by page."""
#     doc = fitz.open(pdf_path)
#     text_by_page = []
#     for page_num, page in enumerate(doc):
#         text_by_page.append((page_num, page.get_text()))
#     return text_by_page

# def split_text_into_chunks(text, chunk_size=300, overlap=50):
#     """Split text into overlapping chunks."""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i : i + chunk_size])
#         chunks.append(chunk)
#     return chunks

# def ingest_documents(data_dir):
#     """Process and ingest all PDFs in the specified directory."""
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".pdf"):
#             pdf_path = os.path.join(data_dir, file_name)
#             text_by_page = extract_text_from_pdf(pdf_path)

#             for page_num, text in text_by_page:
#                 chunks = split_text_into_chunks(text)
#                 for chunk_index, chunk in enumerate(chunks):
#                     store_embedding(file_name, page_num, chunk_index, chunk)
                    
#             print(f"✅ Processed: {file_name}")

# def search_embeddings(query, model_name="minilm", top_k=3):
#     """Retrieve top-k similar chunks using the selected embedding model."""
#     query_embedding, elapsed_time = get_embedding(query, model_name)

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )

#     if not results["metadatas"] or not results["distances"]:
#         print("No relevant results found.")
#         return []

#     top_results = []
#     for metadata_list, distance_list in zip(results["metadatas"], results["distances"]):
#         for metadata, distance in zip(metadata_list, distance_list):
#             top_results.append({
#                 "file": metadata.get("file", "Unknown"),
#                 "page": metadata.get("page", "Unknown"),
#                 "chunk": metadata.get("chunk", "Unknown"),
#                 "text": metadata.get("text", "No text available"),
#                 "similarity": distance
#             })

#     print(f"🔍 Search completed in {elapsed_time:.4f} sec using {model_name}")

#     return top_results[:top_k]

# def generate_rag_response(query, context_results, ollama_model="mistral"):
#     """Generate response using retrieved context."""
#     context_str = "\n".join(
#         [
#             f"From {result.get('file', 'Unknown file')} (Page {result.get('page', 'Unknown page')}, Chunk {result.get('chunk', 'Unknown chunk')}):\n{result.get('text', '')}\n"
#             f"Similarity: {float(result.get('similarity', 0)):.2f}"
#             for result in context_results
#         ]
#     )

#     prompt = f"""You are a helpful AI assistant. 
#     Use the following context to answer the query as accurately as possible. If the context is 
#     not relevant to the query, say 'I don't know'.
    
# Context:
# {context_str}

# Query: {query}

# Answer:"""
    
#     response = ollama.chat(
#         model=ollama_model, messages=[{"role": "user", "content": prompt}]
#     )
    
#     return response["message"]["content"]

# def interactive_search():
#     """Interactive CLI to search documents and get AI-generated responses."""
#     print("\n🔍 **Chroma RAG Search Interface**")
#     print("Type 'exit' to quit\n")
    
#     while True:
#         query = input("\nEnter your search query: ")
#         if query.lower() == "exit":
#             break
        
#         model_choice = input("Choose embedding model (minilm/mpnet/instructorxl): ").strip().lower()
#         if model_choice not in embedding_models:
#             print("Invalid model choice. Using 'minilm' by default.")
#             model_choice = "minilm"

#         context_results = search_embeddings(query, model_name=model_choice)

#         ollama_model_choice = input("Choose Ollama model (mistral/llama3.2): ").strip().lower()
#         if ollama_model_choice not in ["mistral", "llama3.2"]:
#             print("Invalid Ollama model choice. Using 'mistral' by default.")
#             ollama_model_choice = "mistral"

#         response = generate_rag_response(query, context_results, ollama_model_choice)
        
#         print("\n--- Response ---")
#         print(response)

# if __name__ == "__main__":
#     # Ingest documents first
#     file_path = "/Users/CalvinLii/Documents/ds4300/RagIngestAndSearch/data/"
#     print("Ingesting PDFs")
#     ingest_documents(file_path)
#     print("Document Ingestion Complete")

#     # Start the interactive search
#     interactive_search()



import chromadb
import json
import numpy as np
import ollama
import time
import os
import fitz  # PyMuPDF for PDF text extraction
from sentence_transformers import SentenceTransformer

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="ds4300_notes")

# VECTOR_DIM = 768  # Ensure this matches your embedding model

EMBEDDING_VECTOR_DIMS = {
    "nomic": 768,
    "minilm": 384,
    "mpnet": 768
}


# Initialize embedding models
embedding_models = {
    "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    # "instructorxl": SentenceTransformer("hkunlp/instructor-xl")
}

def get_embedding(text: str, model_name: str) -> list:
    """Generate embeddings using the specified model."""
    model = embedding_models.get(model_name)
    # if model is None:
    #     raise ValueError(f"Model {model_name} not found!")

    start_time = time.time()

    if model_name == "nomic":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response["embedding"]
    else:
        model = embedding_models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found!")
        embedding = model.encode(text, normalize_embeddings=True).tolist()


    elapsed_time = time.time() - start_time
    return embedding, elapsed_time

def clear_chroma_store():
    print("Clearing existing ChromaDb  store...")
    chroma_client.clear()
    print("ChromaDb store cleared.")

def store_embedding(collection, file, page, chunk, text, model_name):
    """Store embeddings for multiple models."""
    # embeddings = {}
    # times = {}

    # for model_name in embedding_models.keys():
    #     embedding, elapsed_time = get_embedding(text, model_name)
    #     embeddings[model_name] = embedding
    #     times[model_name] = elapsed_time

    embedding, elapsed_time = get_embedding(text, model_name)
    
    id = f"{file}_page_{page}_chunk_{chunk}"
    
    # Add embeddings only to the embeddings field, while metadata will store scalar values.
    collection.add(
        ids=[id],
        # embeddings=[embeddings["minilm"]],  # Default embedding for indexing
        embeddings = [embedding],
        metadatas=[{
            "file": file,
            "page": page,
            "chunk": chunk,
            "text": text,
            "time": elapsed_time
            # "time_minilm": times["minilm"],
            # "time_mpnet": times["mpnet"],
            # "time_instructorxl": times["instructorxl"]
        }]
    )

    print(f"Stored embedding for: {text}")



def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file by page."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_documents(collection, data_dir, embedding_model):
    """Process and ingest all PDFs in the specified directory."""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    store_embedding(collection, file_name, page_num, chunk_index, chunk, embedding_model)
                    
            print(f"✅ Processed: {file_name}")

def search_embeddings(collection, query, embedding_model="minilm", top_k=5):
    """Retrieve top-k similar chunks using the selected embedding model."""
    query_embedding, elapsed_time = get_embedding(query, embedding_model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results["metadatas"] or not results["distances"]:
        print("No relevant results found.")
        return []

    top_results = []
    for metadata_list, distance_list in zip(results["metadatas"], results["distances"]):
        for metadata, distance in zip(metadata_list, distance_list):
            top_results.append({
                "file": metadata.get("file", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "chunk": metadata.get("chunk", "Unknown"),
                "text": metadata.get("text", "No text available"),
                "similarity": distance
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

# def interactive_search():
#     """Interactive CLI to search documents and get AI-generated responses."""
#     print("\n🔍 **Chroma RAG Search Interface**")
#     print("Type 'exit' to quit\n")
    
#     while True:
#         query = input("\nEnter your search query: ")
#         if query.lower() == "exit":
#             break
        
#         model_choice = input("Choose embedding model (minilm/mpnet/instructorxl): ").strip().lower()
#         if model_choice not in embedding_models:
#             print("Invalid model choice. Using 'minilm' by default.")
#             model_choice = "minilm"

#         context_results = search_embeddings(query, model_name=model_choice)

#         ollama_model_choice = input("Choose Ollama model (mistral/llama3.2): ").strip().lower()
#         if ollama_model_choice not in ["mistral", "llama3.2"]:
#             print("Invalid Ollama model choice. Using 'mistral' by default.")
#             ollama_model_choice = "mistral"

#         response = generate_rag_response(query, context_results, ollama_model_choice)
        
#         print("\n--- Response ---")
#         print(response)

if __name__ == "__main__":
    while True:
            embedding_model= input("Enter the embedding model to use for ingestion and querying (nomic/minilm/mpnet): ").strip().lower()
            if embedding_model in ['nomic','minilm', 'mpnet']:
                vector_dim = EMBEDDING_VECTOR_DIMS[embedding_model]
                break  # Exit the loop when a valid model is entered
            print("Invalid model. Please choose from: nomic, minilm, mpnet.")


   
    # Delete the collection if it exists
    try:
        chroma_client.delete_collection(name="ds4300_notes")
        print("Deleted existing collection 'ds4300_notes'.")
    except: 
        print("doesn't exist")
    # except chromadb.errors.CollectionNotFoundError:
    #     print("Collection 'ds4300_notes' does not exist. Creating a new one.")

    # Create the collection with the correct dimensionality
    collection = chroma_client.create_collection(name="ds4300_notes")

    file_path = "/Users/tristanco/Desktop/DS4300_Practical_2/Data/"
    print("Ingesting PDFs")
    ingest_documents(collection, file_path, embedding_model)
    print("Document Ingestion Complete")


    while True: 
        print(f"\n\n\n----- Quering from previously created collection that uses: {embedding_model}")
        ollama_model_choice = input("Choose Ollama model to provide context to (mistral/llama3.2): ").strip().lower()
        if ollama_model_choice not in ["mistral", "llama3.2"]:
            print("Invalid Ollama model choice. Using 'mistral' by default.")
            ollama_model_choice = "mistral"

        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        context_results = search_embeddings(collection, query, embedding_model)

        response = generate_rag_response(query, context_results, ollama_model_choice)
        
        print("\n--- Response ---")
        print(response)

