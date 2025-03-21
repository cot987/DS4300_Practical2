# import redis
# import json
# import numpy as np
# import ollama
# import time
# from sentence_transformers import SentenceTransformer
# from redis.commands.search.query import Query
# from redis.commands.search.field import VectorField, TextField

# # Initialize models
# embedding_models = {
#     "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
#     "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
#     "instructorxl": SentenceTransformer("hkunlp/instructor-xl")
# }

# redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

# VECTOR_DIM = 1536
# INDEX_NAME = "embedding_index"
# DOC_PREFIX = "doc:"
# DISTANCE_METRIC = "COSINE"

# def get_embedding(text: str, model_name: str) -> list:
#     """Generate embeddings using the specified model."""
#     model = embedding_models.get(model_name)
#     if model is None:
#         raise ValueError(f"Model {model_name} not found!")

#     start_time = time.time()
#     embedding = model.encode(text, normalize_embeddings=True).tolist()
#     elapsed_time = time.time() - start_time

#     return embedding, elapsed_time

# def search_embeddings(query, model_name="minilm", top_k=3):
#     """Search for similar embeddings using Redis."""
#     query_embedding, elapsed_time = get_embedding(query, model_name)
#     query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
    
#     try:
#         q = (
#             Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
#             .sort_by("vector_distance")
#             .return_fields("id", "file", "page", "chunk", "vector_distance")
#             .dialect(2)
#         )

#         start_time = time.time()
#         results = redis_client.ft(INDEX_NAME).search(
#             q, query_params={"vec": query_vector}
#         )
#         search_time = time.time() - start_time

#         top_results = [
#             {
#                 "file": result.file,
#                 "page": result.page,
#                 "chunk": result.chunk,
#                 "similarity": result.vector_distance,
#             }
#             for result in results.docs
#         ][:top_k]


#         # Print results for debugging
#         for result in top_results:
#             print(
#                 f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
#             )
        
#         return top_results, search_time, elapsed_time
    
#     except Exception as e:
#         print(f"Search error: {e}")
#         return [], 0, elapsed_time

# def generate_rag_response(query, context_results, ollama_model="mistral"):
#     """Generate response using context from the search results."""
#     context_str = "\n".join(
#         [
#             f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
#             f"with similarity {float(result.get('similarity', 0)):.2f}"
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
    
#     start_time = time.time()
#     response = ollama.chat(
#         model=ollama_model, messages=[{"role": "user", "content": prompt}]
#     )
#     response_time = time.time() - start_time
    
#     return response["message"]["content"], response_time

# def interactive_search():
#     """Interactive search interface for querying and getting RAG responses."""
#     print("🔍 Redis RAG Search Interface")
#     print("Type 'exit' to quit")
    
#     while True:
#         query = input("\nEnter your search query: ")
#         if query.lower() == "exit":
#             break
        
#         model_choice = input("Choose embedding model (minilm/mpnet/instructorxl): ").strip().lower()
#         if model_choice not in embedding_models:
#             print("Invalid model choice. Using 'minilm' by default.")
#             model_choice = "minilm"
        
#         ollama_model_choice = input("Choose Ollama model (mistral/llama3.2): ").strip().lower()
#         if ollama_model_choice not in ["mistral", "llama3.2"]:
#             print("Invalid Ollama model choice. Using 'mistral' by default.")
#             ollama_model_choice = "mistral"
        
#         # Start the query processing timer
#         start_query_time = time.time()

#         context_results, search_time, embedding_time = search_embeddings(query, model_name=model_choice)
#         response, response_time = generate_rag_response(query, context_results, ollama_model=ollama_model_choice)

#         # Calculate the total query processing time
#         total_query_time = time.time() - start_query_time
        
#         # Print the timings
#         print("\n--- Response ---")
#         print(response)
#         print("\n--- Timings ---")
#         print(f"Embedding time: {embedding_time:.4f} sec")
#         print(f"Search time: {search_time:.4f} sec")
#         print(f"Response generation time: {response_time:.4f} sec")
#         print(f"Total query processing time: {total_query_time:.4f} sec")

# if __name__ == "__main__":
#     interactive_search()

import redis
import json
import numpy as np
import ollama
import time
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField

# Initialize sentence-transformer models
embedding_models = {
    "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 
    # ,"instructorxl": SentenceTransformer("hkunlp/instructor-xl"),
}

redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

# VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def get_embedding(text: str, model_name: str) -> tuple:
    """Generate embeddings using the specified model."""
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

def search_embeddings(query, model_name="nomic", top_k=3):
    """Search for similar embeddings using Redis."""
    query_embedding, elapsed_time = get_embedding(query, model_name)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
    
    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        start_time = time.time()
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )
        search_time = time.time() - start_time

        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        print(results.docs)

        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )
        
        return top_results, search_time, elapsed_time
    
    except Exception as e:
        print(f"Search error: {e}")
        return [], 0, elapsed_time

def generate_rag_response(query, context_results, ollama_model="mistral"):
    """Generate response using context from the search results."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
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
    
    start_time = time.time()
    response = ollama.chat(
        model=f"{ollama_model}:latest", messages=[{"role": "user", "content": prompt}]
    )
    response_time = time.time() - start_time
    
    return response["message"]["content"], response_time

def interactive_search():
    """Interactive search interface for querying and getting RAG responses."""
    print("🔍 Redis RAG Search Interface")
    print("Type 'exit' to quit")
    
    while True:
        metadata_key = "embedding_metadata"
        embedding_name = redis_client.hget(metadata_key, "embedding_model")
        chunk_size = redis_client.hget(metadata_key, "chunk_size")
        overlap = redis_client.hget(metadata_key, "overlap")
        print(f'\nBased on embedding model selected in ingest.py, querying uses {embedding_name}')
        print(f'Using chunk_size: {chunk_size} and overlap: {overlap}')


        # model_choice = input("Choose embedding model (minilm/mpnet/nomic): ").strip().lower() # instructorxl
        # if model_choice not in embedding_models and model_choice != "nomic":
        #     print("Invalid model choice. Using 'nomic' by default.")
        #     model_choice = "nomic"
        
        ollama_model_choice = input("Choose Ollama model (mistral/llama3.2): ").strip().lower()
        if ollama_model_choice not in ["mistral", "llama3.2"]:
            print("Invalid Ollama model choice. Using 'mistral' by default.")
            ollama_model_choice = "mistral"
        
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        # Start the query processing timer
        start_query_time = time.time()

        context_results, search_time, embedding_time = search_embeddings(query, model_name=embedding_name)
        response, response_time = generate_rag_response(query, context_results, ollama_model=ollama_model_choice)

        # Calculate the total query processing time
        total_query_time = time.time() - start_query_time
        
        # Print the timings
        print("\n--- Response ---")
        print(response)
        print("\n--- Timings ---")
        print(f"Embedding time: {embedding_time:.4f} sec")
        print(f"Search time: {search_time:.4f} sec")
        print(f"Response generation time: {response_time:.4f} sec")
        print(f"Total query processing time: {total_query_time:.4f} sec")

if __name__ == "__main__":
    interactive_search()
