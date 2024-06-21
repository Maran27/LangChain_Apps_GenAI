from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = WebBaseLoader("")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
print(retriever)

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever,
                                       "search",
                                       "Search for information")

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [wiki_tool, arxiv_tool, retriever_tool]

from dotenv import load_dotenv
load_dotenv()
import os
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature = 0)

from langchain import hub
prompt = hub.pull("hwchase17/openai-function-agents")
prompt.messages

from langchain.agents import create_openapi_agent
agent = create_openapi_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor

agent_executor.invoke({'input':"Query we need"})