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
        "enqueue-new-forms",
        "Enqueue New Forms",
        ["make", "enqueue_new_forms"],
        description="Add top-N unseen corpus forms to the induction queue",
        cc_ready=True,
    ),
    Action(
        "enqueue-poor-coverage",
        "Enqueue Poor Coverage",
        ["make", "enqueue_poor_coverage"],
        description="Add forms with poor labeled coverage to the induction queue",
        cc_ready=True,
    ),
    Action(
        "induce-senses",
        "Induce Senses",
        ["make", "induce_senses"],
        description="Dequeue forms and run local LLM sense induction",
        cc_ready=False,
    ),
    Action(
        "cc-induce-senses",
        "CC Induce Senses",
        ["make", "cc_induce_senses"],
        description="Dequeue forms and write CC induction task files",
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
            print(f"{a.name:<20} {a.label:<18}  {a.description}")

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
