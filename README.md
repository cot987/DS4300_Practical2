# DS4300 - Spring 2025 - Practical #2 - RAG System 
Authors: Calvin Li, Jona Nakai, Tristan Co 

## High-Level Overview 
Building a local Retrieval-Augmented Generation system that allows a user to query the collective DS 4300 notes. Our system does the following: 
1. Ingest a collection of documents that represent material, such as course notes, you and your team have collected throughout the semester. 
2. Index those documents using embedding and a vector database
3. Accept a query from the user. 
4. Retrieve relevant context based on the user’s query
5. Package the relevant context up into a prompt that is passed to a locally-running LLM to generate a response. 

## DataSet
Collection of class notes compiled from module presentations in https://markfontenot.net/teaching/ds4300/25s-ds4300/ and our own personal notes. 

## Setup Instructions 
## Python Environment
- Your implementation should run with Python 3.11. To create a new conda environment with python 3.11, run the following:

```bash
conda create -n <new_env_name> python=3.11
```

Then activate the newly created python environment with: 
```bash
conda activate <new_env_name> 
```

Subsequently, install additional packages listed in the requirements.txt file with:

```bash
pip install -r requirements.txt
```

## Running Redis Stack 
1. To process the class notes, you have to first run ingest.py and provide the embedding model, chunk size, and overlap. Ensure that process_pdf() in def main() has the correct directory to the data. 
- Examples: `process_pdfs("/path/to/your/dataset", model_name = embedding_name, chunk_size=chunk_size, overlap=overlap)`

2. To query the llm ensure that ollama is running and you Redis Stack container in Docker is running. You can then run redis_db.py.The embedding model inherits from the redis stack embeddings created previously. Select the llm model you intend to search with and provide the query to receive a response. 


## Running Chroma DB and FAISS 
1. Ensure the file_path = in def main() has the correct directory to the data. 
- Examples:`file_path = "/path/to/your/dataset"`

2. There is no separate ingestion process it is directly performed alongside the querying process in the same .py file. Provide the embedding model you intend to use for both ingestion and querying based on the prompt after running either chroma.py or faiss_db.py.

3. Enter LLM you intend to search with and provide the query to receive a response. 


## Short Reflection on Results
Our final recommendation after experimentation and qualitately and quantitatively analyzing result regeneration, the characteristics of the best pipeline has chunking of 500 characters with 50 overlap to provide the most meaningful context. Nomic embedding provides a deatiled and sufficient response with balanced storage. ChromaDB is fast for data ingestion with any sentence transformer, and llama3.2 is the best llm model that provides the most detailed response. Finally, the original prompt that uses role-based prompting and context injection is sufficient and provides a more fleshed out response compared to the modified approach that uses chain-of-thought. 

