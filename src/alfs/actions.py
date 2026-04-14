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


ACTIONS: list[Action] = [
    Action(
        "enqueue-new-forms",
        "Enqueue New Forms",
        ["make", "enqueue_new_forms"],
        description="Add top-N unseen corpus forms to the induction queue",
    ),
    Action(
        "enqueue-poor-coverage",
        "Enqueue Poor Coverage",
        ["make", "enqueue_poor_coverage"],
        description="Add forms with poor labeled coverage to the induction queue",
    ),
    Action(
        "induce-senses",
        "Induce Senses",
        ["make", "induce_senses"],
        description="Dequeue forms and run local LLM sense induction",
    ),
    Action(
        "cc-induce-senses",
        "CC Induce Senses",
        ["make", "cc_induce_senses"],
        description="Dequeue forms and write CC induction task files",
    ),
    Action(
        "clerk",
        "Clerk",
        ["make", "clerk"],
        description="Process queued sense mutations",
    ),
    Action(
        "cc_apply",
        "CC Apply",
        ["make", "cc_apply"],
        description="Apply CC skill outputs as clerk requests",
    ),
    Action(
        "cc-qc",
        "CC Quality Control",
        ["make", "cc_qc"],
        description="Generate CC QC task files for per-form quality control",
    ),
    Action(
        "enqueue-mwe-candidates",
        "Enqueue MWE Candidates",
        ["make", "enqueue_mwe_candidates"],
        description="Compute PMI scores and populate the MWE candidate queue",
    ),
    Action(
        "cc-mwe",
        "CC MWE",
        ["make", "cc_mwe"],
        description="Dequeue MWE candidates and write CC review task files",
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
