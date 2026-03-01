from alfs.data_models.alf import Alf, Sense


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
        for sub in s.subsenses or []:
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


def morph_screen_prompt(forms: list[str]) -> str:
    forms_list = "\n".join(f"  - {f}" for f in forms)
    return (
        "You are a lexicographer identifying morphological derivations.\n"
        "\n"
        "Below is a list of word forms from a dictionary inventory.\n"
        "Identify which forms are likely morphological derivations of a more basic\n"
        "English word (e.g. plural, past tense, comparative, etc.).\n"
        "Only include a form if you believe its base form is a common English word\n"
        "(not a proper noun, not an obscure technical term).\n"
        "\n"
        f"Forms:\n{forms_list}\n"
        "\n"
        "Respond with ONLY valid JSON: "
        '{"candidates": [{"form": "dogs", "base": "dog"}, ...]}\n'
        "If no forms are derivations, return an empty candidates list."
    )


def morph_analyze_prompt(
    derived_form: str,
    base_form: str,
    derived_alf: Alf,
    base_alf: Alf,
) -> str:
    def fmt_senses(alf: Alf) -> str:
        lines = []
        for i, s in enumerate(alf.senses):
            pos_tag = f" [{s.pos}]" if s.pos else ""
            lines.append(f"  {i}. {s.definition}{pos_tag}")
        return "\n".join(lines) if lines else "  (no senses)"

    return (
        "You are a lexicographer identifying morphological relationships between\n"
        "dictionary sense pairs.\n"
        "\n"
        f'Derived form: "{derived_form}"\n'
        f"Senses (0-indexed):\n{fmt_senses(derived_alf)}\n"
        "\n"
        f'Base form: "{base_form}"\n'
        f"Senses (0-indexed):\n{fmt_senses(base_alf)}\n"
        "\n"
        "For each sense of the derived form that is a direct morphological derivation\n"
        "of a sense of the base form, provide:\n"
        "  - derived_sense_idx: index into derived form's senses\n"
        "  - base_sense_idx: index into base form's senses\n"
        "  - relation: one of plural, past_tense, third_person_singular,\n"
        "    present_participle, comparative, superlative\n"
        "  - proposed_definition: a concise dictionary-style definition\n"
        '    (e.g. "plural form of dog (n.)")\n'
        "\n"
        'Respond with ONLY valid JSON: {"relations": [{"derived_sense_idx": 0,\n'
        '"base_sense_idx": 1, "relation": "plural",\n'
        '"proposed_definition": "plural form of dog (n.)"}]}\n'
        "If no derivational links exist, return an empty relations list."
    )


def critic_prompt(form: str, before: list[Sense], after: list[Sense]) -> str:
    lines = [
        "You are a senior lexicographer reviewing a proposed revision"
        " to a dictionary entry.",
        "",
        f'Word: "{form}"',
        "",
        "Original definitions:",
    ]
    for i, s in enumerate(before, 1):
        lines.append(f"  {i}. {s.definition}")
        for sub in s.subsenses or []:
            lines.append(f"     \u2022 {sub}")
    lines += ["", "Proposed definitions:"]
    for i, s in enumerate(after, 1):
        lines.append(f"  {i}. {s.definition}")
        for sub in s.subsenses or []:
            lines.append(f"     \u2022 {sub}")
    lines += [
        "",
        "Is the proposed version an improvement over the original?",
        "Reject if any proposed definition:",
        "  - is self-referential (uses the word or a close derivative of it)",
        "  - is too terse (no more informative than the original)",
        "  - is too verbose or padded with unnecessary hedging",
        "  - changes the level of granularity without good reason",
        "  - is otherwise worse as a dictionary entry",
        "",
        'Respond with ONLY valid JSON: {"is_improvement": true, "reason": "..."}',
    ]
    return "\n".join(lines)


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
