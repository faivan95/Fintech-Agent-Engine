import os
from langchain_ollama import ChatOllama

ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Used for ideation and risk assessment
llm_creative = ChatOllama(
    base_url=ollama_base_url,
    model="phi3:mini",
    temperature=0.7, 
    keep_alive="60s" 
)

# Used strictly for writing syntactically perfect Python logic
llm_coder = ChatOllama(
    base_url=ollama_base_url,
    model="phi3:mini",
    temperature=0.0, 
    keep_alive="60s" 
)