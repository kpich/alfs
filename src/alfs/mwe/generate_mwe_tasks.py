"""Generate cc-mwe task files from the MWE candidate queue.

Dequeues candidates, finds corpus occurrences, extracts context windows,
and writes CCMWETask JSON files to cc_tasks/pending/mwe/.

Usage:
    python -m alfs.mwe.generate_mwe_tasks \
        --mwe-queue-file ../alfs_data/mwe_queue.yaml \
        --seg-data-dir by_prefix/ --docs docs.parquet \
        --cc-tasks-dir ../cc_tasks [--n 20] [--max-contexts 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import uuid

import polars as pl

from alfs.data_models.mwe_queue import MWEQueue
from alfs.data_models.occurrence import Occurrence
from alfs.encoding import context_window as _context_window
from alfs.mwe.find_occurrences import (
    find_mwe_occurrences,
    load_all_seg_data,
    mwe_form_from_components,
)


def run(
    mwe_queue_file: Path,
    seg_data_dir: Path,
    docs: Path,
    cc_tasks_dir: Path,
    *,
    n: int = 20,
    max_contexts: int = 8,
    context_chars: int = 150,
    seed: int | None = 42,
) -> int:
    """Generate up to n cc-mwe task files. Returns count generated."""
    queue = MWEQueue(mwe_queue_file)
    entries = queue.dequeue(n)
    if not entries:
        print("MWE queue is empty.")
        return 0

    pending_dir = cc_tasks_dir / "pending" / "mwe"
    pending_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading seg data from {seg_data_dir}...")
    all_tokens = load_all_seg_data(seg_data_dir)

    rng = random.Random(seed)
    generated = 0

    for entry in entries:
        # Find occurrences if not already present
        if entry.occurrences:
            occs = entry.occurrences
        else:
            occs = find_mwe_occurrences(all_tokens, entry.components)

        if not occs:
            print(f"  skipped {entry.form!r}: no occurrences found")
            continue

        # Sample occurrences for context extraction
        sampled = rng.sample(occs, min(max_contexts, len(occs)))

        # Load needed docs
        needed_doc_ids = list({o.doc_id for o in sampled})
        docs_df = (
            pl.scan_parquet(str(docs))
            .filter(pl.col("doc_id").is_in(needed_doc_ids))
            .collect(engine="streaming")
        )
        docs_map = dict(
            zip(
                docs_df["doc_id"].to_list(),
                docs_df["text"].to_list(),
                strict=False,
            )
        )

        # Extract contexts
        form = mwe_form_from_components(entry.components)
        contexts: list[str] = []
        occurrence_refs: list[Occurrence] = []
        for occ in sampled:
            text = docs_map.get(occ.doc_id, "")
            if not text:
                continue
            snippet, _ = _context_window(text, occ.byte_offset, form, context_chars)
            contexts.append(snippet)
            occurrence_refs.append(occ)

        if not contexts:
            print(f"  skipped {entry.form!r}: no contexts extracted")
            continue

        task = {
            "type": "mwe",
            "id": str(uuid.uuid4()),
            "form": entry.form,
            "components": entry.components,
            "pmi": entry.pmi,
            "corpus_count": entry.corpus_count,
            "contexts": contexts,
            "occurrence_refs": [o.model_dump() for o in occurrence_refs],
        }

        out_path = pending_dir / f"{task['id']}.json"
        out_path.write_text(json.dumps(task, indent=2))
        print(f"  wrote {entry.form!r} → {out_path.name}")
        generated += 1

    print(f"Generated {generated} cc-mwe task(s).")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cc-mwe task files from MWE candidate queue"
    )
    parser.add_argument("--mwe-queue-file", required=True)
    parser.add_argument("--seg-data-dir", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--cc-tasks-dir", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        Path(args.mwe_queue_file),
        Path(args.seg_data_dir),
        Path(args.docs),
        Path(args.cc_tasks_dir),
        n=args.n,
        max_contexts=args.max_contexts,
        context_chars=args.context_chars,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
