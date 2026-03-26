def labeling_prompt(form: str, context: str, sense_menu: str) -> str:
    """Return the occurrence-labeling prompt for Ollama."""
    return (
        f'The word "{form}" appears here: "...{context}..."\n'
        f"\n"
        f'Senses of "{form}":\n'
        f"{sense_menu}\n"
        f"\n"
        f'Which sense applies? Use "1", "2", "1a", "1b", etc.\n'
        f"Rate applicability: 2=excellent, 1=okay (needs a more refined sense),"
        f" 0=poor/doesn't fit.\n"
        f'Also list synonyms: other words that could roughly fit in place of "{form}"'
        f" here. Doesn't need to be a perfect match — approximate or related words"
        f" are fine. Use [] if nothing fits at all.\n"
        f"\n"
        f'Respond with ONLY valid JSON: {{"sense_key": "1", "rating": 2,'
        f' "synonyms": ["word1", "word2"]}}\n'
        f'(If rating is 0, set sense_key to "0".)'
    )
