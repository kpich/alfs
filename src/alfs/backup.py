"""Backup senses.db to letter-bucketed YAML files in a sibling repo.

Usage:
    python -m alfs.backup \\
        --senses-db ../alfs_data/senses.db --senses-repo ../alfs_senses \\
        [--queue-dir ../clerk_queue]
"""

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess

from pydantic import TypeAdapter
import yaml

from alfs.clerk.request import ChangeRequest
from alfs.data_models.sense_store import SenseStore

_adapter: TypeAdapter[ChangeRequest] = TypeAdapter(ChangeRequest)


def write_mutation_log(queue_dir: Path, senses_repo: Path) -> None:
    """Append new clerk mutations from done/ to monthly JSONL files in senses repo."""
    done_dir = queue_dir / "done"
    if not done_dir.exists():
        return

    # Load IDs already committed to the log
    mutations_dir = senses_repo / "mutations"
    logged_ids: set[str] = set()
    if mutations_dir.exists():
        for jsonl_file in mutations_dir.glob("*.jsonl"):
            for line in jsonl_file.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            logged_ids.add(data["id"])
                    except json.JSONDecodeError:
                        pass

    # Collect new entries from done/
    new_entries: list[ChangeRequest] = []
    for json_file in sorted(done_dir.glob("*.json")):
        try:
            request = _adapter.validate_json(json_file.read_bytes())
            req_id = getattr(request, "id", None)
            if req_id is None or req_id not in logged_ids:
                new_entries.append(request)
        except Exception:
            pass

    if not new_entries:
        print("  No new mutations to log.")
        return

    # Group by YYYY-MM based on created_at
    groups: dict[str, list[ChangeRequest]] = defaultdict(list)
    for req in new_entries:
        created_at = getattr(req, "created_at", None)
        if isinstance(created_at, datetime):
            month_key = created_at.strftime("%Y-%m")
        else:
            month_key = "unknown"
        groups[month_key].append(req)

    mutations_dir.mkdir(exist_ok=True)
    _epoch = datetime(1970, 1, 1, tzinfo=UTC)

    def _sort_key(r: ChangeRequest) -> datetime:
        dt = getattr(r, "created_at", _epoch) or _epoch
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    for month_key, entries in sorted(groups.items()):
        entries.sort(key=_sort_key)
        out_path = mutations_dir / f"{month_key}.jsonl"
        with out_path.open("a") as f:
            for entry in entries:
                f.write(_adapter.dump_json(entry).decode() + "\n")
        print(f"  Appended {len(entries)} mutations → mutations/{month_key}.jsonl")


def backup_yaml_files(
    senses_repo: Path,
    blocklist_file: Path | None,
    queue_file: Path | None,
    mwe_skipped_file: Path | None = None,
) -> None:
    """Copy YAML support files into senses_repo root."""
    import shutil

    for src in [blocklist_file, queue_file, mwe_skipped_file]:
        if src is not None and src.exists():
            dst = senses_repo / src.name
            shutil.copy2(src, dst)
            print(f"  Copied {src.name} → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup senses to YAML")
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--senses-repo", required=True, help="Path to sibling senses git repo"
    )
    parser.add_argument(
        "--queue-dir", help="Path to clerk queue dir (enables mutation log)"
    )
    parser.add_argument(
        "--blocklist-file", default=None, help="Path to blocklist.yaml (optional)"
    )
    parser.add_argument(
        "--queue-file", default=None, help="Path to induction_queue.yaml (optional)"
    )
    parser.add_argument(
        "--mwe-skipped-file", default=None, help="Path to mwe_skipped.yaml (optional)"
    )
    args = parser.parse_args()

    store = SenseStore(Path(args.senses_db))
    entries = store.all_entries()
    timestamps = store.all_timestamps()

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
        entry_dict = alf.model_dump(exclude_none=True, mode="json")
        ts = timestamps.get(form)
        if ts:
            entry_dict["updated_at"] = ts
        buckets.setdefault(bucket_key, []).append(entry_dict)

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

    if args.queue_dir:
        write_mutation_log(Path(args.queue_dir), repo)

    backup_yaml_files(
        repo,
        Path(args.blocklist_file) if args.blocklist_file else None,
        Path(args.queue_file) if args.queue_file else None,
        Path(args.mwe_skipped_file) if args.mwe_skipped_file else None,
    )

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
