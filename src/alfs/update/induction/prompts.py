def induction_prompt(
    form: str, contexts: list[str], existing_defs: list[str] | None = None
) -> str:
    """Return the sense-induction prompt for Ollama."""
    numbered = "\n".join(f"{i + 1}. {ctx}" for i, ctx in enumerate(contexts))
    n = len(contexts)

    existing_block = ""
    opt_out_clause = (
        f'Find the single most common meaning of "{form}" in these sentences'
        f" that is NOT already covered above.\n"
    )
    if existing_defs:
        numbered_defs = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(existing_defs))
        existing_block = (
            f"This word already has these senses defined:\n" f"{numbered_defs}\n" f"\n"
        )
        opt_out_clause = (
            f"If ALL of these sentences are already well covered by the existing"
            f" senses above, "
            f'output {{"all_covered": true, "definition": "",'
            f' "examples": [], "subsenses": []}}.\n'
            f'Otherwise, find the single most common meaning of "{form}"'
            f" in these sentences that is NOT already covered above.\n"
        )

    return (
        f'You are a lexicographer. Below are {n} sentences containing "{form}".\n'
        f"{existing_block}"
        f"{opt_out_clause}"
        f"Group the sentence numbers that illustrate it,"
        f" write a concise one-sentence definition.\n"
        f"Output ONLY a JSON object — no prose, no markdown, no explanation.\n"
        f"Start your response with {{ and end with }}.\n"
        f"\n"
        f"Sentences:\n"
        f"{numbered}\n"
        f"\n"
        f'{{"all_covered": false, "definition": "...", "examples": [1, 2],'
        f' "subsenses": []}}'
    )
