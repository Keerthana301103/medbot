from langchain.llms import Ollama
from config import OLLAMA_MODEL

def get_ollama_response(prompt):
    llm = Ollama(model=OLLAMA_MODEL)
    response = llm(prompt)
    return response
