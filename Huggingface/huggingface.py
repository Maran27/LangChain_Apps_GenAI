from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import numpy as np

loader = PyPDFDirectoryLoader("./Folder_name")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_document = text_splitter.split_documents(docs)

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings' : True}
)
np.array(huggingface_embeddings.embed_query(final_document[0].page_content))
print(np.array(huggingface_embeddings.embed_query(final_document[0].page_content)))
print(np.array(huggingface_embeddings.embed_query(final_document[0].page_content)).shape)

vectorstore=FAISS.from_documents(final_document[:120],huggingface_embeddings)
query="WHAT IS HEALTH INSURANCE COVERAGE?"
relevant_docments=vectorstore.similarity_search(query)
print(relevant_docments[0].page_content)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
print(retriever)

import os
os.environ['HUGGINGFACEHUB_API_TOKEN']=""

from langchain_community.llms import HuggingFaceHub
hf=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1,"max_length":500}

)
query="What is the health insurance coverage?"
hf.invoke(query)

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
hf = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-v0.1",
    task="text-generation",
    pipeline_kwargs={"temperature": 0, "max_new_tokens": 300}
)
llm = hf
llm.invoke(query)

prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context
{context}
Question:{question}
Helpful Answers:
"""

prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
retrievalQA=RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)

query="""DIFFERENCES IN THE
UNINSURED RATE BY STATE
IN 2022"""
result = retrievalQA.invoke({"query": query})
print(result['result'])


