"""LLM-assisted redirect detection for case-variant entries.

Usage:
    python -m alfs.update.dedupe \\
        --alfs ../alfs_data/alfs.json --output ../alfs_data/alfs.json \\
        [--model gemma2:9b] [--dry-run]
"""

import argparse
from pathlib import Path

from alfs.data_models.alf import Alf, Alfs
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


def find_candidates(alfs: Alfs) -> list[tuple[str, str]]:
    """Return (form, lower_form) pairs that are redirect candidates."""
    candidates = []
    for form, alf in alfs.entries.items():
        lower = form.lower()
        if lower == form:
            continue
        if lower not in alfs.entries:
            continue
        if alf.redirect is not None:
            continue
        if alfs.entries[lower].redirect is not None:
            continue
        candidates.append((form, lower))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-assisted redirect detection for case-variant entries"
    )
    parser.add_argument("--alfs", required=True, help="Path to alfs.json")
    parser.add_argument("--output", required=True, help="Output path for alfs.json")
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print verdicts without modifying alfs.json",
    )
    args = parser.parse_args()

    alfs = Alfs.model_validate_json(Path(args.alfs).read_text())
    candidates = find_candidates(alfs)

    if not candidates:
        print("No candidates found.")
        return

    print(f"Found {len(candidates)} candidate(s).")
    entries = dict(alfs.entries)

    for form, lower_form in candidates:
        form_defs = [s.definition for s in alfs.entries[form].senses]
        lower_defs = [s.definition for s in alfs.entries[lower_form].senses]

        prompt = prompts.dedup_prompt(form, form_defs, lower_form, lower_defs)
        data = llm.chat_json(args.model, prompt, format=_DEDUP_SCHEMA)

        is_redirect = bool(data.get("is_redirect", False))
        reason = data.get("reason", "")

        verdict = "REDIRECT" if is_redirect else "keep"
        print(f"  {form!r} -> {lower_form!r}: {verdict} — {reason}")

        if is_redirect and not args.dry_run:
            existing = entries[form]
            entries[form] = Alf(
                form=existing.form,
                senses=list(existing.senses),
                redirect=lower_form,
            )

    if not args.dry_run:
        updated = Alfs(entries=entries)
        Path(args.output).write_text(updated.model_dump_json(indent=2))
        print(f"Wrote updated inventory to {args.output}")
    else:
        print("[dry-run] No changes written.")


if __name__ == "__main__":
    main()
