"""Backup senses.db to letter-bucketed YAML files in a sibling repo.

Usage:
    python -m alfs.backup \\
        --senses-db ../alfs_data/senses.db --senses-repo ../alfs_senses
"""

import argparse
from pathlib import Path
import subprocess

import yaml

from alfs.data_models.sense_store import SenseStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup senses to YAML")
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--senses-repo", required=True, help="Path to sibling senses git repo"
    )
    args = parser.parse_args()

    store = SenseStore(Path(args.senses_db))
    entries = store.all_entries()

    buckets: dict[str, list[dict]] = {}
    for form, alf in sorted(entries.items()):
        first = form[0].lower() if form and form[0].isalpha() else None
        key = first if first is not None else "special"
        buckets.setdefault(key, []).append(alf.model_dump(exclude_none=True))

    repo = Path(args.senses_repo)
    repo.mkdir(parents=True, exist_ok=True)

    for key, alfs_list in sorted(buckets.items()):
        out_path = repo / f"{key}.yaml"
        out_path.write_text(yaml.dump(alfs_list, allow_unicode=True, sort_keys=False))
        print(f"  Wrote {len(alfs_list)} entries → {out_path}")

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "backup senses"],
        cwd=repo,
        check=True,
    )
    print(f"Committed to {repo}")


if __name__ == "__main__":
    main()
