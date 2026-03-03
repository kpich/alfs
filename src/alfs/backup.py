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

    # buckets keyed by (dir, file) where dir=first letter, file=digraph (first two)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for form, alf in sorted(entries.items()):
        first = form[0].lower() if form and form[0].isalpha() else None
        if first is None:
            bucket_key: tuple[str, str] = ("", "special")
        else:
            second = form[1].lower() if len(form) > 1 and form[1].isalpha() else ""
            digraph = first + second
            bucket_key = (first, digraph)
        buckets.setdefault(bucket_key, []).append(
            alf.model_dump(exclude_none=True, mode="json")
        )

    repo = Path(args.senses_repo)
    repo.mkdir(parents=True, exist_ok=True)

    for (dir_key, file_key), alfs_list in sorted(buckets.items()):
        if not dir_key:
            out_path = repo / "special.yaml"
        else:
            out_dir = repo / dir_key
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{file_key}.yaml"
        out_path.write_text(yaml.dump(alfs_list, allow_unicode=True, sort_keys=False))
        print(f"  Wrote {len(alfs_list)} entries → {out_path}")

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    result = subprocess.run(
        ["git", "commit", "-m", "backup senses"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Committed to {repo}")
        subprocess.run(["git", "push"], cwd=repo, check=True)
        print("Pushed.")
    else:
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            print("Nothing new to commit.")
        else:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )


if __name__ == "__main__":
    main()
