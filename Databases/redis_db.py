import redis
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

# Global client and variables 
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

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

def search_embeddings(query, model_name="nomic", top_k=5):
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
        # print(results.docs)
        print("\n\n--- Context --- ")
        for result in top_results:
            print(
                f"\n---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )
        
        return top_results, search_time, elapsed_time
    
    except Exception as e:
        print(f"Search error: {e}")
        return [], 0, elapsed_time

def generate_rag_response(query, context_results, ollama_model="mistral"):
    """Generate response using context from the search results."""
    start_time = time.time()
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )


    # For multi-step questions 
    # prompt = f""" If the query contains multiple sub-questions or concepts, break it down into parts: 
    # 1. Identify each sub-question. 
    # 2. First, extract key concepts and facts from the query to understand the intent. 
    # Next, carefully analyze the retrieved context to identify the most relevant details. 
    # Summarize these findings concisely in bullet points, ensuring that the information is directly related to the query.
    # 3. Answer each part separately before creating a final response. 

    # # Using chain of thought 
    # prompt = f"""
    # You are a helpful AI assistant. 

    # First, extract key concepts and facts from the query to understand the intent. 
    # Next, carefully analyze the retrieved context to identify the most relevant details for the query. 
    # Explain these findings in bullet points, ensuring that the information is directly related to the query.
    # Then, create an answer using the context. 
    # If the context is not relevant to the query, say  "I don't know."

    # Structure your response as follows:
    # 1. Extracted Key Concepts: [list of key terms] 

    # 2. Context:
    # - [bullet point]
    # - [bullet point]

    # 3. Final Answer: \n[response]

    # Using role directives (persona-based)
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
    response_time = time.time() - start_time
    
    return response["message"]["content"], response_time

def interactive_search():
    """Interactive search interface for querying and getting RAG responses."""
    print("🔍 Redis RAG Search Interface")
    print("Type 'exit' to quit")
    
    while True:

        # Finds the appropriate embedding model to use for querying and chunk overlap metadata 
        metadata_key = "embedding_metadata"
        embedding_name = redis_client.hget(metadata_key, "embedding_model")
        chunk_size = redis_client.hget(metadata_key, "chunk_size")
        overlap = redis_client.hget(metadata_key, "overlap")
        print(f"\n--- Redis Query using previously ingested: {embedding_name}---")
        print(f'Using chunk_size: {chunk_size} and overlap: {overlap}')

        # Input for llm model 
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
        print(f"Query embedding time: {embedding_time:.4f} sec")
        print(f"Search time: {search_time:.4f} sec")
        print(f"Response generation time: {response_time:.4f} sec")
        print(f"Total query time: {total_query_time:.4f} sec")

if __name__ == "__main__":
    interactive_search()
