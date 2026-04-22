import os
from langchain_ollama import ChatOllama

# We retrieve the host URL from the environment variables set in docker-compose.yml.
# If it's missing, we fallback to localhost for local debugging.
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Initialize the ChatOllama model. 
# We are using phi3 as it is highly efficient and fits your 6GB VRAM constraint.
llm = ChatOllama(
    base_url=ollama_base_url,
    model="phi3:mini",
    temperature=0.7, # 0.7 allows for creativity in strategy generation, but keeps it grounded
    # HARDWARE OPTIMIZATION: Tell Ollama to keep the model in VRAM for only 60 seconds 
    # instead of 5 minutes after generating, freeing up your GTX 1060 for other tasks.
    keep_alive="60s"
)