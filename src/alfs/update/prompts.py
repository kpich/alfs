def induction_prompt(
    form: str, contexts: list[str], existing_defs: list[str] | None = None
) -> str:
    """Return the sense-induction prompt for Ollama."""
    numbered = "\n".join(f"{i + 1}. {ctx}" for i, ctx in enumerate(contexts))
    n = len(contexts)

    existing_block = ""
    if existing_defs:
        numbered_defs = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(existing_defs))
        existing_block = (
            f"This word already has these senses defined:\n" f"{numbered_defs}\n" f"\n"
        )

    return (
        f'You are a lexicographer. Below are {n} sentences containing "{form}".\n'
        f"{existing_block}"
        f'Find the single most common meaning of "{form}" in these sentences'
        f" that is NOT already covered above.\n"
        f"Group the sentence numbers that illustrate it,"
        f" write a concise one-sentence definition.\n"
        f"Output ONLY a JSON object — no prose, no markdown, no explanation.\n"
        f"Start your response with {{ and end with }}.\n"
        f"\n"
        f"Sentences:\n"
        f"{numbered}\n"
        f"\n"
        f'{{"definition": "...", "examples": [1, 2], "subsenses": []}}'
    )


def dedup_prompt(
    form: str,
    form_defs: list[str],
    lower_form: str,
    lower_defs: list[str],
) -> str:
    def fmt_defs(defs: list[str]) -> str:
        if not defs:
            return "  (no definitions yet)"
        return "\n".join(f"  {i+1}. {d}" for i, d in enumerate(defs))

    return (
        f"You are a lexicographer deciding whether a capitalized word form is merely\n"
        f"a sentence-initial variant of its lowercase counterpart"
        f" with no distinct meaning.\n"
        f"\n"
        f'Capitalized form: "{form}"\n'
        f"Its definitions:\n{fmt_defs(form_defs)}\n"
        f"\n"
        f'Lowercase form: "{lower_form}"\n'
        f"Its definitions:\n{fmt_defs(lower_defs)}\n"
        f"\n"
        f'Is "{form}" simply "{lower_form}" capitalized at sentence start, '
        f"with no distinct meaning of its own?\n"
        f'\nRespond with ONLY valid JSON: {{"is_redirect": true, "reason": "..."}}'
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
        f"Rate applicability: 3=excellent, 2=reasonable, 1=poor,"
        f" 0=none of the listed senses apply.\n"
        f"\n"
        f'Respond with ONLY valid JSON: {{"sense_key": "1", "rating": 3}}\n'
        f'(If rating is 0, set sense_key to "0".)'
    )
