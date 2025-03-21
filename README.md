# DS4300_Practical_2

1. Make sure to install all required packages (chromadb, redis, faiss, etc.)
2. For running redis_db.py, first run ingest.py
    - For both these files, make sure to change directory path to the location of the Data folder you want ingested
3. For running chroma.py, change directory path to the location of the Data folder you want to ingest
4. For running faiss_db.py, also change the directory to your Data folder.
5. Running the script for any database will first ingest the desired documents. Then it will ask user for the embedding model to use. Lastly, it will ask for the llm model to construct a response, and return a response after asking a question query.
