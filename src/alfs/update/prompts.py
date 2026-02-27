def induction_prompt(form: str, contexts: list[str]) -> str:
    """Return the sense-induction prompt for Ollama."""
    numbered = "\n".join(f"{i + 1}. {ctx}" for i, ctx in enumerate(contexts))
    return (
        f"You are a lexicographer. Here are example sentences containing the word"
        f'"{form}".\nIdentify its distinct senses. For each, write a concise '
        f"one-sentence definition.\n"
        f"Include subsenses only when clearly warranted.\n"
        f"\n"
        f"Examples:\n"
        f"{numbered}\n"
        f"\n"
        f"Respond with ONLY valid JSON (no markdown):\n"
        f'{{"senses": [{{"definition": "...", "subsenses": ["..."]}}]}}'
    )


def labeling_prompt(form: str, context: str, sense_menu: str) -> str:
    """Return the occurrence-labeling prompt for Ollama."""
    return (
        f'The word "{form}" appears here: "...{context}..."\n'
        f"\n"
        f'Senses of "{form}":\n'
        f"{sense_menu}\n"
        f"\n"
        f'Which sense applies? Use "1", "2", "1a", "1b", etc.\n'
        f"Rate applicability: 3=excellent, 2=reasonable, 1=poor.\n"
        f"\n"
        f'Respond with ONLY valid JSON: {{"sense_key": "1", "rating": 3}}'
    )
