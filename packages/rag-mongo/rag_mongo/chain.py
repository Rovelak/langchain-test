import os

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pymongo import MongoClient


# Set DB
if os.environ.get("MONGO_URI", None) is None:
    raise Exception("Missing `MONGO_URI` environment variable.")
MONGO_URI = os.environ["MONGO_URI"]

DB_NAME = "langchain-test-2"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

SEARCH_K_VALUE = 50
POST_FILTER_PIPELINE_LIMIT = 5
OPENAI_MODEL_NAME = "gpt-3.5-turbo-16k-0613"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]

# Read from MongoDB Atlas Vector Search
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

#retriever = vectorstore.as_retriever(search_type="mmr")

# retriever = vectorstore.as_retriever(
#   search_type="similarity_score_threshold", search_kwargs={"score_threshold": -0.2, 
#   "post_filter_pipeline": [{"$project": { "_id": 0}}]}
# )
retriever = vectorstore.as_retriever(
  search_type="similarity", search_kwargs={"k": SEARCH_K_VALUE, 
  "post_filter_pipeline": [{"$project": { "_id": 0}}]}
)

# RAG prompt
template = """Answer the question based only on the following context:
{context}
More context: one full day is 7h of work, corresponding to a 100 percent claim
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0)

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)