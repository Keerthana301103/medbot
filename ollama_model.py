import ollama

def chat_with_ollama(prompt):
    """Generates a response using Ollama's LLM."""
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
