# Import libraries 
import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import os
import fitz
import time
from memory_profiler import profile

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

# Global variables 
EMBEDDING_VECTOR_DIMS = {
    "nomic": 768,
    "minilm": 384,
    "mpnet": 768
}

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


def clear_redis_store():
    """used to clear the redis vector store """
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


def create_hnsw_index(VECTOR_DIM):
    """ Create an HNSW index in Redis """
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


def get_embedding(text: str, model_name: str = "nomic") -> list:
    """ Generate embeddings """

    if model_name == "nomic": 
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response["embedding"]
    elif model_name == "minilm":
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(text, normalize_embeddings=True).tolist()
    elif model_name == "mpnet":
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embedding = model.encode(text, normalize_embeddings=True).tolist()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return embedding


def store_embedding(file: str, page: str, chunk: str, embedding: list, embedding_model: str):
    """ 
    store the embedding in Redis
    More robust key doc_prefix, filename, page, chunk 
    Takes any pdf file and extracts by page and chunks it 
    """
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding_model": embedding_model, 

            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")

def store_embedding_metadata(embedding_model: str, chunk_size:int, overlap:int):
    """ 
    custom embedding for metadata 
    """
    metadata_key = f"embedding_metadata"
    
    # Metadata for which embedding model to use for querying 
    redis_client.hset(
        metadata_key,
        mapping={
            "embedding_model": embedding_model,
            "chunk_size" : chunk_size, 
            "overlap": overlap
        },
    )
    print(f"Stored metadata for")

 
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

@profile
def process_pdfs(data_dir, model_name ="nomic", chunk_size=300, overlap=50):
    """ Process all PDF files in a given directory """

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            store_embedding_metadata(model_name, chunk_size, overlap)

            for page_num, text in text_by_page:
                # Chunking the text certain number of characters for vectorizing 
                chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk, model_name)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding_model= model_name, 
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def query_redis(query_text: str, model_name = "nomic"):
    """ query redis by finding the 5 nearest embeddings from redis store """
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text, model_name)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


def main():
    print("\n--- Ingestion ---")
    
    # Inputs 
    while True:
        embedding_name = input("Enter the embedding model to use for ingestion and querying (nomic/minilm/mpnet): ").strip().lower()
        if embedding_name in EMBEDDING_VECTOR_DIMS:
            vector_dim = EMBEDDING_VECTOR_DIMS[embedding_name]
            break  
        print("Invalid model. Please choose from: nomic, minilm, mpnet.")
    vector_dim = EMBEDDING_VECTOR_DIMS[embedding_name]

    chunk_size = int(input("Enter chunk size:"))
    overlap = int(input("Enter overlap size:"))

    # Processing pdf files 
    clear_redis_store()
    start_index_time = time.time()
    create_hnsw_index(vector_dim)
    process_pdfs("/Users/CalvinLii/Documents/ds4300/DS4300_Practical_2/data/", model_name = embedding_name, chunk_size=chunk_size, overlap=overlap)

    index_time = time.time() - start_index_time

    # Timing data
    print(f"Ingestion time: {index_time:.4f} sec")
    print(f"\n---Done processing PDFs using {embedding_name}---\n")


if __name__ == "__main__":
    main()



