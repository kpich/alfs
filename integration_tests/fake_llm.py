"""Fake LLM for integration tests — replaces llm.chat_json via monkeypatch."""


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, model: str, prompt: str, retries: int = 3, format=None) -> dict:
        self.calls.append((model, prompt))
        return self.responses.pop(0)
