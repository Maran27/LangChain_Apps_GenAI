from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI # to create a chat application
from langserve import add_routes # using this routes we can interact with multiple FMs
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model = ChatOpenAI()
llm = Ollama(model="gemma")

prompt1 = ChatPromptTemplate.from_template("Write a code for this {topic} in python")
prompt2 = ChatPromptTemplate.from_template("Write a code for this {topic} in Javascript")

add_routes(
    app,
    prompt1|model,
    path="/python"
)

add_routes(
    app,
    prompt2|llm,
    path="/JS"
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000
    )
