def induction_critic_prompt(
    form: str, definition: str, existing_defs: list[str]
) -> str:
    existing_block = ""
    if existing_defs:
        numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(existing_defs))
        existing_block = f"Existing senses:\n{numbered}\n\n"
    return (
        f"This is a broad-coverage English language dictionary — broader than a "
        f'traditional dictionary. It includes individual letters (e.g. "D", "K"), '
        f"abbreviations, acronyms, symbols, slang, technical jargon, and other tokens "
        f"with recognized public meaning, even if they would not appear in a "
        f"conventional dictionary. "
        f'Review this proposed dictionary sense for "{form}".\n'
        f"{existing_block}"
        f'Proposed definition: "{definition}"\n\n'
        f"Default to accepting. Only reject if you are highly confident of "
        f"one of these:\n"
        f'- "{form}" is a parser artifact, OCR error, or has no recognized meaning '
        f"(garbled tokens, stray punctuation, malformed sequences with no known use). "
        f"Rare, archaic, obscure, or non-standard forms with real uses are fine.\n"
        f'- "{form}" is a foreign word that would not appear in an English '
        f"dictionary — it occurs almost entirely in non-English text rather than "
        f"as a loanword or expression commonly used in English. "
        f"Genuine loanwords (café, résumé, schadenfreude) are fine.\n"
        f"- The definition is factually wrong or nonsensical (not merely imprecise "
        f"or general — imprecise definitions are acceptable).\n"
        f'- The definition is circular: it uses "{form}" itself (or an obvious direct '
        f'derivative like "{form}s" or "{form}ed") as the core of the explanation, '
        f'e.g. "animals: the state of being an animal". Using related vocabulary '
        f"is normal dictionary practice and is NOT circular.\n"
        f"- The proposed meaning substantially overlaps with an existing sense — "
        f"covering the same core usage even if worded differently or at a different "
        f"level of specificity.\n"
        f"- The definition describes a meaning that only exists as part of a specific "
        f"fixed multi-word expression or idiom, rather than a standalone word sense "
        f'(e.g. rejecting a sense of "a" for its role in "a priori").\n\n'
        f"Words commonly have multiple distinct senses — a new sense being different "
        f"from existing ones is expected and good, not a reason to reject. "
        f'For example, "through" meaning "by means of" is a valid new sense even if '
        f'an existing sense covers "movement from one side to the other".\n\n'
        f"Do NOT reject for vagueness, generality, or imprecision. "
        f"When in doubt, accept.\n\n"
        f'Output JSON only: {{"is_valid": true, "reason": "..."}}'
    )


def induction_prompt(
    form: str, contexts: list[str], existing_defs: list[str] | None = None
) -> str:
    """Return the sense-induction prompt for Ollama."""
    numbered = "\n".join(f"{i + 1}. {ctx}" for i, ctx in enumerate(contexts))
    n = len(contexts)

    existing_block = ""
    opt_out_clause = (
        f'If "{form}" is a parsing artifact with no recognized meaning (garbled tokens,'
        f" stray punctuation), or if it is a foreign word occurring almost entirely in "
        f"non-English text rather than as a loanword or expression used in English, "
        f'output {{"all_covered": true, "senses": []}}.\n'
        f'Otherwise, find all major distinct meanings of "{form}"'
        f" clearly represented in these sentences.\n"
    )
    if existing_defs:
        numbered_defs = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(existing_defs))
        existing_block = (
            f"This word already has these senses defined:\n" f"{numbered_defs}\n" f"\n"
        )
        opt_out_clause = (
            f"If all sentences are already covered by the existing senses, or if "
            f'"{form}" is a parsing artifact with no recognized meaning (garbled '
            f"tokens, stray punctuation), or if it is a foreign word occurring almost "
            f"entirely in non-English text rather than as a loanword or expression "
            f"used in English, "
            f'output {{"all_covered": true, "senses": []}}.\n'
            f'Otherwise, find all major distinct meanings of "{form}"'
            f" in these sentences that are NOT already covered above.\n"
        )

    return (
        f"You are a lexicographer working on a broad-coverage English dictionary — "
        f"broader than a traditional dictionary. It includes individual letters "
        f'(e.g. "D", "K"), abbreviations, acronyms, symbols, slang, technical jargon, '
        f"and other tokens with recognized public meaning.\n"
        f'Below are {n} example sentences containing "{form}"'
        f" (a sample — there may be many other uses not shown).\n"
        f"{existing_block}"
        f"{opt_out_clause}"
        f"Only include a sense if the sentences clearly attest that meaning as a"
        f" general English usage — not if it is merely plausible, and not if the"
        f" usage appears specific to a single document's context rather than a"
        f" general word sense.\n"
        f"Write definitions that describe the word's general English meaning —"
        f" not narrowly tailored to the specific example sentences shown.\n"
        f"Each sense must be meaningfully distinct — not paraphrasable as another.\n"
        f"Do not include a sense that is only meaningful as part of a specific fixed "
        f'multi-word expression or idiom (e.g. do not add a sense of "a" for its role '
        f'in "a priori"). If the word participates in many idioms or phrasal verbs as '
        f"a component, you may write one generic sense describing that pattern (e.g. "
        f'"used as a component in many idiomatic expressions and phrasal verbs") '
        f"rather than listing individual MWE meanings.\n"
        f"For each sense, group the sentence numbers that illustrate it,"
        f" write a concise one-sentence definition, and assign a part of speech.\n"
        f"Choose pos from: noun, verb, adjective, adverb, preposition, conjunction,"
        f" pronoun, determiner, interjection, proper_noun, other.\n"
        f"Output ONLY a JSON object — no prose, no markdown, no explanation.\n"
        f"Start your response with {{ and end with }}.\n"
        f"\n"
        f"Sentences:\n"
        f"{numbered}\n"
        f"\n"
        f'{{"all_covered": false, "senses": [{{"definition": "...", "examples": [1, 2],'
        f' "pos": "noun"}}]}}'
    )
