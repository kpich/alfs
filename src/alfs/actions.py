from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys


@dataclass
class Action:
    name: str
    label: str
    cmd: list[str]
    description: str = ""
    cc_ready: bool = (
        False  # True if this action works in CC mode (has CC branch or no LLM)
    )


ACTIONS: list[Action] = [
    Action(
        "update",
        "Update",
        ["make", "update"],
        description="Run ETL update pipeline",
        cc_ready=True,
    ),
    Action(
        "relabel",
        "Relabel",
        ["make", "relabel"],
        description="Re-run labeling pipeline",
    ),
    Action(
        "label_new",
        "Label New",
        ["make", "label_new"],
        description="Label new corpus instances for random forms from inventory",
    ),
    Action("dedupe", "Dedupe", ["make", "dedupe"], description="Deduplicate senses"),
    Action(
        "cleanup",
        "Cleanup",
        ["make", "cleanup"],
        description="Clean up sense inventory",
        cc_ready=True,
    ),
    Action(
        "rewrite",
        "Rewrite",
        ["make", "rewrite"],
        description="Rewrite sense definitions",
        cc_ready=True,
    ),
    Action(
        "retag",
        "Retag",
        ["make", "retag"],
        description="Retag occurrences",
    ),
    Action(
        "prune",
        "Prune",
        ["make", "prune"],
        description="Prune low-quality senses",
        cc_ready=True,
    ),
    Action(
        "spelling_variant",
        "Spelling Variant",
        ["make", "spelling_variant"],
        description="Find British/American spelling variant pairs",
        cc_ready=True,
    ),
    Action(
        "morph_redirect",
        "Morph Redirect",
        ["make", "morph_redirect"],
        description="Propose morphological derivation links",
        cc_ready=True,
    ),
    Action(
        "undo_morph",
        "Undo Morph",
        ["make", "undo_morph"],
        description="Detect and undo incorrect morphological links",
    ),
    Action(
        "trim_senses",
        "Trim Senses",
        ["make", "trim_senses"],
        description="Trim redundant senses",
        cc_ready=True,
    ),
    Action(
        "delete_entry",
        "Delete Entry",
        ["make", "delete_entry"],
        description="Remove mistokenized/artifact word entries",
        cc_ready=True,
    ),
    Action(
        "clerk",
        "Clerk",
        ["make", "clerk"],
        description="Process queued sense mutations",
        cc_ready=True,
    ),
    Action(
        "cc_apply",
        "CC Apply",
        ["make", "cc_apply"],
        description="Apply CC skill outputs as clerk requests",
        cc_ready=True,
    ),
]

ACTIONS_BY_NAME: dict[str, Action] = {a.name: a for a in ACTIONS}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m alfs.actions <list|run> [name]", file=sys.stderr)
        sys.exit(1)

    cmd = args[0]

    if cmd == "list":
        for a in ACTIONS:
            print(f"{a.name:<10} {a.label:<12}  {a.description}")

    elif cmd == "run":
        if len(args) < 2:
            print("Usage: python -m alfs.actions run <name>", file=sys.stderr)
            sys.exit(1)
        name = args[1]
        action = ACTIONS_BY_NAME.get(name)
        if action is None:
            print(f"Unknown action: {name!r}", file=sys.stderr)
            print(f"Available: {', '.join(ACTIONS_BY_NAME)}", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run(action.cmd)
        sys.exit(result.returncode)

    else:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
