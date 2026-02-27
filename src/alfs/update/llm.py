import ollama


def chat(model: str, prompt: str) -> str:
    """Send a single-turn chat to the given Ollama model and return the text."""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
