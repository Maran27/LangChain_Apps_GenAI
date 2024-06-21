# First multiple data ingestion steps.

from langchain_community.document_loaders import TextLoader
loader = TextLoader("speech.txt")
text_documents = loader.load()
print(text_documents)

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader
import bs4
loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/post/2023-06-23-agent/",),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=("post-title","post-content","post-header")
                       )),)
text_documents1=loader.load()

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('abc.pdf')
documents = loader.load()

# TRansform

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents1 = text_splitter.split_documents(documents)
documents1[:5]

# Vector Embedding and Vector Store Database

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(documents[:20], OpenAIEmbeddings)

query = ""
result = db.similarity_search(query)
result[0].page_content

from langchain_community.vectorstores import FAISS
db1 = FAISS.from_documents(documents[:20], OpenAIEmbeddings())

query = ""
result = db1.similarity_search(query)
result[0].page_content
