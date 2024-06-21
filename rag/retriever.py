from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('abc.pdf')
documents = loader.load()
print(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents1 = text_splitter.split_documents(documents)
print(documents1[:5])

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
db1 = FAISS.from_documents(documents[:20], OpenAIEmbeddings())
query = ""
result = db1.similarity_search(query)
print(result[0].page_content)

from langchain_community.llms import Ollama
llm = Ollama('gemma')
print(llm)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following questions based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}""")

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = db1.as_retriever()

from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

result = retrieval_chain.invoke({'imput':''})
result['answer']
