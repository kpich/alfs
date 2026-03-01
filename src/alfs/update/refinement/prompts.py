from alfs.data_models.alf import Sense


def rewrite_prompt(form: str, senses: list[Sense]) -> str:
    lines = [
        "You are a lexicographer improving dictionary entries.",
        "",
        f'Rewrite the sense definitions for "{form}" to be clearer and more precise.',
        "Keep exactly the same number of senses. Preserve meaning; improve phrasing.",
        "",
        "Current definitions:",
    ]
    for i, s in enumerate(senses, 1):
        lines.append(f"  {i}. {s.definition}")
        for sub in s.subsenses:
            lines.append(f"     \u2022 {sub}")
    lines += [
        "",
        "Respond with ONLY valid JSON: "
        '{"senses": [{"definition": "...", "subsenses": [...]}, ...]}',
    ]
    return "\n".join(lines)


def postag_prompt(form: str, definition: str, instances: list[str]) -> str:
    instances_section = ""
    if instances:
        numbered = "\n".join(
            f"  {i + 1}. ...{ctx}..." for i, ctx in enumerate(instances)
        )
        instances_section = f"\nExample uses in context:\n{numbered}\n"

    return (
        f"You are a lexicographer assigning a part-of-speech tag to a dictionary sense."
        f"\n\n"
        f'Word: "{form}"\n'
        f"Definition: {definition}\n"
        f"{instances_section}"
        f"\nWhat is the primary part of speech for this sense?\n"
        f"Choose one of: noun, verb, adjective, adverb, preposition, conjunction, "
        f"pronoun, determiner, interjection, other.\n"
        f'\nRespond with ONLY valid JSON: {{"pos": "noun"}}'
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
