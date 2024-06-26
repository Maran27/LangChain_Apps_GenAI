from dotenv import load_dotenv
import os
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader
loader=WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
data = loader.load()

from langchain_openai import OpenAIEmbeddings
from langchain_objectbox.vectorstores import ObjectBox
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=text_splitter.split_documents(data)

vector = ObjectBox.from_documents(docs, OpenAIEmbeddings(), embedding_dimension=768)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub

llm = ChatOpenAI(model='gpt-4o')
prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = vector.as_retriever(),
    chain_type_kwargs={"prompt":prompt}
)

question=""
result = qa_chain({"query":question})
result['result']

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(result['result'])
