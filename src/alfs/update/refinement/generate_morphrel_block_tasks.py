"""Generate CC task files for morph-rel / redirect / block decisions.

For each sampled wordform, writes a CCMorphRelBlockTask JSON file to
cc_tasks/pending/morphrel_block/. Forms that are already redirects, fully
morph-tagged, blocklisted, or already have a pending/done task file are skipped.

Usage:
    python -m alfs.update.refinement.generate_morphrel_block_tasks \\
        --senses-db ../alfs_data/senses.db \\
        --cc-tasks-dir ../cc_tasks \\
        [--blocklist-file ../alfs_data/blocklist.yaml] \\
        [--n 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import uuid

from alfs.cc.models import CCMorphRelBlockTask, SenseInfo
from alfs.data_models.blocklist import Blocklist
from alfs.data_models.sense_store import SenseStore


def run(
    senses_db: str | Path,
    cc_tasks_dir: str | Path,
    n: int = 20,
    blocklist_file: str | Path | None = None,
) -> None:
    tasks_dir = Path(cc_tasks_dir)
    pending_dir = tasks_dir / "pending" / "morphrel_block"
    done_dir = tasks_dir / "done" / "morphrel_block"
    pending_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    # Collect form strings that already have a task file (pending or done).
    existing_forms: set[str] = set()
    for task_file in list(pending_dir.glob("*.json")) + list(done_dir.glob("*.json")):
        try:
            import json

            data = json.loads(task_file.read_bytes())
            if isinstance(data.get("form"), str):
                existing_forms.add(data["form"])
        except Exception:
            pass

    blocklist: Blocklist | None = None
    if blocklist_file:
        blocklist = Blocklist(Path(blocklist_file))

    sense_store = SenseStore(Path(senses_db))
    all_entries = sense_store.all_entries()

    eligible = []
    for alf in all_entries.values():
        if not alf.senses:
            continue
        if all(s.morph_base is not None for s in alf.senses):
            continue
        if blocklist is not None and blocklist.contains(alf.form):
            continue
        if alf.form in existing_forms:
            continue
        eligible.append(alf)

    if not eligible:
        print("No eligible forms found.")
        return

    sample = random.sample(eligible, min(n, len(eligible)))

    for alf in sample:
        task = CCMorphRelBlockTask(
            id=str(uuid.uuid4()),
            form=alf.form,
            senses=[
                SenseInfo(
                    id=s.id,
                    definition=s.definition,
                    pos=str(s.pos.value) if s.pos else None,
                )
                for s in alf.senses
            ],
        )
        path = pending_dir / f"{task.id}.json"
        path.write_text(task.model_dump_json())
        print(f"  wrote task for {alf.form!r} → {path.name}")

    print(f"Generated {len(sample)} task file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CC morphrel_block task files"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--cc-tasks-dir", required=True, help="Path to cc_tasks directory"
    )
    parser.add_argument(
        "--blocklist-file", default=None, help="Path to blocklist.yaml (optional)"
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Number of forms to sample (default: 20)"
    )
    args = parser.parse_args()

    run(
        senses_db=args.senses_db,
        cc_tasks_dir=args.cc_tasks_dir,
        n=args.n,
        blocklist_file=args.blocklist_file,
    )


if __name__ == "__main__":
    main()
