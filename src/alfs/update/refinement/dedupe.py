"""LLM-assisted redirect detection for case-variant entries.

Usage:
    python -m alfs.update.refinement.dedupe \\
        --senses-db ../alfs_data/senses.db \\
        [--model gemma2:9b] [--dry-run]
"""

import argparse
from pathlib import Path

from alfs.data_models.alf import Alf
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

_DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_redirect": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_redirect", "reason"],
}


def find_candidates(entries: dict[str, Alf]) -> list[tuple[str, str]]:
    """Return (form, lower_form) pairs that are redirect candidates."""
    candidates = []
    for form, alf in entries.items():
        lower = form.lower()
        if lower == form:
            continue
        if lower not in entries:
            continue
        if alf.redirect is not None:
            continue
        if entries[lower].redirect is not None:
            continue
        candidates.append((form, lower))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-assisted redirect detection for case-variant entries"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print verdicts without modifying senses.db",
    )
    args = parser.parse_args()

    store = SenseStore(Path(args.senses_db))
    entries = store.all_entries()
    candidates = find_candidates(entries)

    if not candidates:
        print("No candidates found.")
        return

    print(f"Found {len(candidates)} candidate(s).")

    for form, lower_form in candidates:
        form_defs = [s.definition for s in entries[form].senses]
        lower_defs = [s.definition for s in entries[lower_form].senses]

        prompt = prompts.dedup_prompt(form, form_defs, lower_form, lower_defs)
        data = llm.chat_json(args.model, prompt, format=_DEDUP_SCHEMA)

        is_redirect = bool(data.get("is_redirect", False))
        reason = data.get("reason", "")

        verdict = "REDIRECT" if is_redirect else "keep"
        print(f"  {form!r} -> {lower_form!r}: {verdict} — {reason}")

        if is_redirect and not args.dry_run:
            existing = entries[form]
            store.write(
                Alf(
                    form=existing.form,
                    senses=list(existing.senses),
                    redirect=lower_form,
                )
            )

    if args.dry_run:
        print("[dry-run] No changes written.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
