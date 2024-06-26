from langchain_huggingface import HuggingFacePipeline
#llm = HuggingFacePipeline(pipeline='text-generation', model='google/flan-t5-xxl')
llm = HuggingFacePipeline.from_model_id(
    model_id="",
    task='text-generation',
    pipeline_kwargs = {"temperature":0.1, "max_length":100, "top_k":50},
)
llm.invoke("question")

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

from langchain_huggingface import HuggingFaceEndpoint
llm1 = HuggingFaceEndpoint(
    repo_id="",
    task='text-generation',
    max_new_tokens=100,
    do_sample=False,
)
llm1.invoke("Question")
