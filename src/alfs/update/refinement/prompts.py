from collections.abc import Sequence

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
        pos_tag = f" [{s.pos.value}]" if s.pos else ""
        lines.append(f"  {i}.{pos_tag} {s.definition}")
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
        f"pronoun, determiner, interjection, proper_noun, other.\n"
        f'\nRespond with ONLY valid JSON: {{"pos": "noun"}}'
    )


def postag_critic_prompt(
    form: str, definition: str, pos: str, instances: list[str]
) -> str:
    instances_section = ""
    if instances:
        numbered = "\n".join(
            f"  {i + 1}. ...{ctx}..." for i, ctx in enumerate(instances)
        )
        instances_section = f"\nExample uses in context:\n{numbered}\n"

    return (
        f"You are a lexicographer verifying a part-of-speech tag assignment.\n"
        f"\n"
        f'Word: "{form}"\n'
        f"Definition: {definition}\n"
        f"{instances_section}"
        f"\nProposed POS: {pos}\n"
        f"\nIs this POS tag correct for this sense?\n"
        f'\nRespond with ONLY valid JSON: {{"is_valid": true, "reason": "..."}}'
    )


def morph_screen_prompt(forms: list[str]) -> str:
    forms_list = "\n".join(f"  - {f}" for f in forms)
    return (
        "You are a lexicographer identifying morphological derivations.\n"
        "\n"
        "Below is a list of word forms from a dictionary inventory.\n"
        "Identify forms that are regular inflections of a more basic English word.\n"
        "Qualifying categories:\n"
        "  - Plural forms (dogs ← dog, boxes ← box)\n"
        "  - Verbal inflections (walked ← walk, running ← run, goes ← go)\n"
        "  - Comparative or superlative forms (faster ← fast, best ← good)\n"
        "\n"
        "Do NOT include:\n"
        "  - Pronoun case changes (our ← we, him ← he, her ← she)\n"
        "  - Derivational morphology (happiness ← happy, driver ← drive)\n"
        "  - Semantic or etymological relations (second ← same)\n"
        "  - A form that is its own base\n"
        "\n"
        "Only include a form if its base is a common English word"
        " (not a proper noun, not an obscure term).\n"
        "If in doubt, leave it out.\n"
        "\n"
        f"Forms:\n{forms_list}\n"
        "\n"
        "Respond with ONLY valid JSON: "
        '{"candidates": [{"form": "dogs", "base": "dog"}, ...]}\n'
        "If no forms qualify, return an empty candidates list."
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
        "  - relation: a short description of the inflectional relationship\n"
        '    (e.g. "plural", "verbal inflection", "comparative form")\n'
        "  - proposed_definition: a concise dictionary-style definition\n"
        '    (e.g. "plural form of dog (n.)")\n'
        "\n"
        'Respond with ONLY valid JSON: {"relations": [{"derived_sense_idx": 0,\n'
        '"base_sense_idx": 1, "relation": "plural",\n'
        '"proposed_definition": "plural form of dog (n.)"}]}\n'
        "If no derivational links exist, return an empty relations list."
    )


def trim_sense_prompt(form: str, senses: list[Sense], examples: list[list[str]]) -> str:
    lines = [
        "You are a lexicographer reviewing dictionary senses for redundancy.",
        "",
        f'Word: "{form}"',
        "",
        "Senses:",
    ]
    for i, (s, exs) in enumerate(zip(senses, examples, strict=False), 1):
        lines.append(f"  {i}. {s.definition}")
        for sub in s.subsenses or []:
            lines.append(f"     \u2022 {sub}")
        for ex in exs:
            lines.append(f"     \u2014 {ex}")
    lines += [
        "",
        "Should any sense be deleted? Delete if two senses cover the same concept"
        " and one is weaker, or if the form is a parsing artifact rather than a"
        " real word or expression.",
        "",
        "If so, give the sense NUMBER (1-based) to delete and a brief reason.",
        "If all senses are worth keeping, set sense_num to null.",
        "",
        'Respond with ONLY valid JSON: {"sense_num": 2, "reason": "..."}',
    ]
    return "\n".join(lines)


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
        pos_tag = f" [{s.pos.value}]" if s.pos else ""
        lines.append(f"  {i}.{pos_tag} {s.definition}")
        for sub in s.subsenses or []:
            lines.append(f"     \u2022 {sub}")
    lines += ["", "Proposed definitions:"]
    for i, s in enumerate(after, 1):
        pos_tag = f" [{s.pos.value}]" if s.pos else ""
        lines.append(f"  {i}.{pos_tag} {s.definition}")
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


def morph_critic_prompt(
    derived_form: str,
    base_form: str,
    relation: str,
    proposed_definition: str,
) -> str:
    return (
        "You are a lexicographer verifying a proposed morphological redirect.\n"
        "\n"
        f'Derived form: "{derived_form}"\n'
        f'Base form: "{base_form}"\n'
        f"Proposed relation: {relation}\n"
        f"Proposed definition: {proposed_definition}\n"
        "\n"
        f'Is "{derived_form}" a genuine regular inflectional form of "{base_form}"'
        " with the described relationship?\n"
        "Regular inflections include plurals, verbal inflections (past tense,"
        " present participle, etc.), and comparative/superlative forms.\n"
        "Reject if:\n"
        "  - the relationship is pronoun case (our ← we), derivational"
        " (happiness ← happy), or semantic/etymological\n"
        "  - the forms are not morphologically related at all\n"
        "  - the proposed definition does not accurately describe the link\n"
        "\n"
        'Respond with ONLY valid JSON: {"is_valid": true, "reason": "..."}'
    )


def undo_morph_screen_prompt(items: Sequence[tuple[str, int, Sense]]) -> str:
    lines = [
        "You are a lexicographer auditing a dictionary for incorrect"
        " morphological links.",
        "",
        "Below are numbered dictionary senses that have been tagged as regular"
        " inflections of a base word.",
        "Identify any links that are clearly wrong — the form is NOT a genuine regular"
        " inflection (plural, verbal inflection, comparative/superlative) of the base.",
        "Also flag cases where the definition does not match the claimed link.",
        "",
        "For each bad link, propose a replacement standalone definition that describes"
        " the form's actual meaning without reference to a base form.",
        "If all links look correct, return an empty bad_links list.",
        "",
        "Items:",
    ]
    for i, (form, sense_idx, sense) in enumerate(items, 1):
        morph_base = getattr(sense, "morph_base", None) or "?"
        morph_relation = getattr(sense, "morph_relation", None) or "inflection"
        definition = getattr(sense, "definition", "")
        lines.append(f'{i}. "{form}" (sense {sense_idx})')
        lines.append(f"   Definition: {definition}")
        lines.append(f'   Morph link: {morph_relation} of "{morph_base}"')
    lines += [
        "",
        "Respond with ONLY valid JSON:",
        '{"bad_links": [{"item_num": 1, "proposed_definition": "..."}, ...]}',
    ]
    return "\n".join(lines)


def undo_morph_critic_prompt(
    form: str,
    sense_idx: int,
    morph_base: str,
    morph_relation: str,
    old_def: str,
    new_def: str,
) -> str:
    return (
        "You are a lexicographer reviewing a proposed correction"
        " to a dictionary entry.\n"
        "\n"
        f'Word: "{form}" (sense {sense_idx})\n'
        f"Current definition: {old_def}\n"
        f'Morph link: tagged as "{morph_relation}" of "{morph_base}"\n'
        "\n"
        f"Proposed action: remove the morph link and replace the definition with:\n"
        f'  "{new_def}"\n'
        "\n"
        f"Should the morph link be removed? Approve if the link is genuinely wrong"
        f' (not a regular inflection of "{morph_base}") and the proposed'
        f" standalone definition is reasonable.\n"
        "Reject if the morph link is actually valid or the proposed definition is"
        " poor quality.\n"
        "\n"
        'Respond with ONLY valid JSON: {"is_valid": true, "reason": "..."}'
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
