def labeling_prompt(form: str, context: str, sense_menu: str) -> str:
    """Return the occurrence-labeling prompt for Ollama."""
    return (
        f'The word "{form}" appears here: "...{context}..."\n'
        f"\n"
        f'Senses of "{form}":\n'
        f"{sense_menu}\n"
        f"\n"
        f'Which sense applies? Use "1", "2", "1a", "1b", etc.\n'
        f"Rate applicability: 2=excellent match for this sense, 1=okay but could be"
        f" more specific, 0=the word is NOT being used in any of these senses (e.g.,"
        f" it's a different meaning entirely, it's being used as a proper name, it's"
        f" slang for something else). When in doubt between 0 and 1, use 0.\n"
        f'Also list synonyms: other words that could roughly fit in place of "{form}"'
        f" here. Doesn't need to be a perfect match — approximate or related words"
        f" are fine. Use [] if nothing fits at all.\n"
        f"\n"
        f'Respond with ONLY valid JSON: {{"sense_key": "1", "rating": 2,'
        f' "synonyms": ["word1", "word2"]}}\n'
        f"If rating is 0, choose sense_key as follows:\n"
        f'  - "0" if the occurrence is noise, OCR garbage, a parsing artifact,'
        f" a proper noun, or a truly one-off usage that should never be"
        f" re-examined for sense assignment.\n"
        f'  - "_none" if the word is being used in some real meaning that none'
        f" of these senses covers (a new sense may be warranted)."
    )
