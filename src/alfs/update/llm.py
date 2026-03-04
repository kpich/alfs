import json
import re
from typing import Any

import anthropic as _anthropic_sdk
import ollama

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Per-call timeout in seconds; prevents threads from hanging indefinitely if
# ollama becomes unresponsive.
_TIMEOUT = 600

_client = ollama.Client(timeout=_TIMEOUT)

_anthropic_client: _anthropic_sdk.Anthropic | None = None


def _anthropic() -> _anthropic_sdk.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = _anthropic_sdk.Anthropic()
    return _anthropic_client


def chat(model: str, prompt: str, format: dict[str, Any] | None = None) -> str:
    """Send a single-turn chat to the given model and return the text.

    If model starts with 'claude-', routes to Anthropic API; otherwise Ollama.
    The format arg is unused for Claude (JSON instructions are embedded in prompts).
    """
    if model.startswith("claude-"):
        msg = _anthropic().messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        block = msg.content[0]
        assert isinstance(block, _anthropic_sdk.types.TextBlock)
        return block.text
    response = _client.chat(
        model=model, messages=[{"role": "user", "content": prompt}], format=format
    )
    return response["message"]["content"]


def _scan_json_objects(text: str) -> list[dict]:  # type: ignore[type-arg]
    """Scan text for all valid top-level JSON objects."""
    decoder = json.JSONDecoder()
    objects = []
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = text.find("{", end)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
    return objects


def chat_json(
    model: str, prompt: str, retries: int = 3, format: dict[str, Any] | None = None
) -> dict:  # type: ignore[type-arg]
    """Send a chat and parse the JSON response, with retries.

    Tries in order: code fences → full parse → scan for embedded JSON objects.
    """
    last_err: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        raw = chat(model, prompt, format=format).strip()

        # Try code fences
        m = _JSON_FENCE_RE.search(raw)
        candidate = m.group(1).strip() if m else raw
        try:
            return json.loads(candidate)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Scan for any embedded JSON object; prefer ones with expected keys
        objects = _scan_json_objects(raw)
        if objects:
            for obj in reversed(objects):
                if "senses" in obj or "sense_key" in obj:
                    return obj  # type: ignore[return-value]
            return objects[0]  # type: ignore[return-value]

        print(f"[llm] attempt {attempt + 1}/{retries} — no JSON found in: {raw!r}")
        last_err = json.JSONDecodeError("no JSON found in response", raw, 0)
    raise last_err
