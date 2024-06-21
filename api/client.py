# this is my app.

import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/python/invoke",
                             json={'input':{'topic':input_text}})
    return response.json()['output']['content']

def get_gemma_response(input_text):
    response = requests.post("http://localhost:8000/JS/invoke",
                             json={'input':{'topic':input_text}})
    return response.json()['output']

st.title("Langchain Demo with API")
input_text = st.text_input("Write the code in python for ")
input_text1 = st.text_input("Write the code in javascript for ")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_gemma_response(input_text1))
