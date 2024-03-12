import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

MONGO_URI = os.environ["MONGO_URI"]

# Note that if you change this, you also need to change it in `rag_mongo/chain.py`
DB_NAME = "langchain-test-2"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"
EMBEDDING_FIELD_NAME = "embedding"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]
# PDF file to split location
FILE_URI = "/home/rovelak/Documents/claims" 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 10

def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        # Extract the chunk
        chunks.append(text[start:end])
        # Move start and end forward, considering the overlap
        start = end - chunk_overlap
        end = start + chunk_size
        if end > len(text):
            # Ensure the last chunk extends to the end of the text
            end = len(text)
    
    return chunks

if __name__ == "__main__":
    # Load docs
    data  = load_text_file(FILE_URI)

    # Split docs
    docs = split_text_into_chunks(data, CHUNK_SIZE, CHUNK_OVERLAP)

    # Insert the documents in MongoDB Atlas Vector Search
    _ = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
