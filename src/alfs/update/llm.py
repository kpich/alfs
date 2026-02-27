import json
import re

import ollama

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def chat(model: str, prompt: str) -> str:
    """Send a single-turn chat to the given Ollama model and return the text."""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def chat_json(model: str, prompt: str, retries: int = 3) -> dict:  # type: ignore[type-arg]
    """Send a chat and parse the JSON response, stripping markdown fences, with
    retries."""
    last_err: Exception = RuntimeError("no attempts made")
    for _ in range(retries):
        text = chat(model, prompt).strip()
        m = _JSON_FENCE_RE.search(text)
        if m:
            text = m.group(1).strip()
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            last_err = e
    raise last_err
