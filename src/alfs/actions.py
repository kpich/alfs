from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys


@dataclass
class Action:
    name: str
    label: str
    cmd: list[str]
    human_review: bool = False
    description: str = ""


ACTIONS: list[Action] = [
    Action(
        "update", "Update", ["make", "update"], description="Run ETL update pipeline"
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
        "postag", "POS-tag", ["make", "postag"], description="Add part-of-speech tags"
    ),
    Action(
        "cleanup",
        "Cleanup",
        ["make", "cleanup"],
        description="Clean up sense inventory",
    ),
    Action(
        "rewrite",
        "Rewrite",
        ["make", "rewrite"],
        human_review=True,
        description="Rewrite sense definitions (requires approval)",
    ),
    Action(
        "retag",
        "Retag",
        ["make", "retag"],
        human_review=True,
        description="Retag occurrences (requires approval)",
    ),
    Action(
        "prune",
        "Prune",
        ["make", "prune"],
        human_review=True,
        description="Prune low-quality senses (requires approval)",
    ),
    Action(
        "morph_redirect",
        "Morph Redirect",
        ["make", "morph_redirect"],
        human_review=True,
        description="Propose morphological derivation links (requires approval)",
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
            h = " [H]" if a.human_review else ""
            print(f"{a.name:<10} {a.label:<12}{h}  {a.description}")

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
