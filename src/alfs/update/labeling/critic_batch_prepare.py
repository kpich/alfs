"""Prepare a Groq critic batch to validate sense labels.

For each (form, sense) pair with enough labeled instances, a critic LLM sees a
numbered list of occurrence contexts and identifies which ones do NOT match the
sense definition. Instances flagged as bad are later downgraded to rating=0 by
critic_batch_ingest; every reviewed instance also gets a last_critic_date stamp.

Usage:
    python -m alfs.update.labeling.critic_batch_prepare \\
        --senses-db senses.db --labeled-db labeled.db --docs docs.parquet \\
        --output-dir critic_batch/ \\
        [--instances-per-sense 20] [--min-instances 5] \\
        [--model openai/gpt-oss-20b] [--context-chars 150] \\
        [--seed 42] [--max-batch-size 50000]

Output files in --output-dir:
    critic_input_{batch_id}_{NNN:03d}.jsonl    Groq batch API input
    critic_metadata_{batch_id}_{NNN:03d}.jsonl  Sidecar with instance lists
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import polars as pl

from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.reserved_sense_keys import RESERVED_SENSE_KEYS
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.label_occurrences import extract_context

# Models confirmed to support response_format={"type":"json_object"} on Groq.
# Omit from this set if unsure — the prompt already enforces JSON output.
_JSON_MODE_MODELS: frozenset[str] = frozenset(
    {
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
    }
)


def build_critic_system_message(form: str, definition: str) -> str:
    """System message for the critic prompt (cached per form+sense by Groq)."""
    return (
        f"You are reviewing word-sense labels for a dictionary.\n"
        f"\n"
        f'The word "{form}" is supposed to be used in this sense:\n'
        f'"{definition}"\n'
        f"\n"
        f'I will show you numbered uses of "{form}". For each, decide: is this '
        f"occurrence best described by the sense defined above?\n"
        f"\n"
        f"Mark an occurrence bad if it does not match — including when the word is "
        f"clearly being used in a different sense (even if the definition above could "
        f"loosely apply), not just when the definition fails entirely.\n"
        f"\n"
        f'Respond ONLY with valid JSON: {{"bad_indices": [1, 3]}} listing numbers of '
        f"examples that do NOT best match the sense above. Use [] if all match."
    )


def run(
    senses_db: str | Path,
    labeled_db: str | Path,
    docs: str | Path,
    output_dir: str | Path,
    instances_per_sense: int = 20,
    min_instances: int = 5,
    model: str = "llama-3.3-70b-versatile",
    context_chars: int = 150,
    seed: int | None = None,
    max_batch_size: int = 50_000,
    max_senses: int | None = None,
    batch_id: str | None = None,
) -> list[tuple[Path, Path]]:
    """Build critic batch input and metadata JSONL files in output_dir.

    Returns list of (batch_path, metadata_path) pairs, one per chunk.
    """
    if batch_id is None:
        batch_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    sense_store = SenseStore(Path(senses_db))
    occ_store = OccurrenceStore(Path(labeled_db))

    # Build sense_uuid → (form, definition) lookup from all entries
    sense_info: dict[str, tuple[str, str]] = {}
    for form, alf in sense_store.all_entries().items():
        for sense in alf.senses:
            if sense.definition:
                sense_info[sense.id] = (form, sense.definition)

    # max sense updated_at per form for staleness detection
    max_sense_ts = sense_store.max_sense_updated_at_by_form()

    # Load all rating >= 1 instances from labeled.db
    labeled = occ_store.to_polars()
    if len(labeled) == 0:
        print("No labeled instances found.")
        return []

    good = labeled.filter(pl.col("rating").is_in([1, 2]))

    # Group instances by (sense_uuid, form, definition), keeping only those
    # needing critic review: last_critic_date IS NULL or stale vs latest sense edit
    # Each entry: (doc_id, byte_offset, surface_form)
    groups: dict[tuple[str, str, str], list[tuple[str, int, str]]] = {}
    for row in good.iter_rows(named=True):
        sense_uuid = str(row["sense_key"])
        if sense_uuid in RESERVED_SENSE_KEYS or sense_uuid not in sense_info:
            continue
        form, definition = sense_info[sense_uuid]
        last_critic: str | None = row["last_critic_date"]  # type: ignore[assignment]
        max_ts = max_sense_ts.get(form)
        needs_review = last_critic is None or (
            max_ts is not None and last_critic < max_ts
        )
        if not needs_review:
            continue
        key = (sense_uuid, form, definition)
        groups.setdefault(key, []).append(
            (str(row["doc_id"]), int(row["byte_offset"]), str(row["form"]))  # type: ignore[arg-type]
        )

    # Filter to senses with enough instances needing review
    eligible = {k: v for k, v in groups.items() if len(v) >= min_instances}
    print(
        f"Found {len(eligible)} senses with >= {min_instances} instances needing review"
    )

    # Sample up to instances_per_sense per sense
    rng = np.random.default_rng(seed)
    sampled: list[tuple[str, str, str, list[tuple[str, int, str]]]] = []
    for (sense_uuid, form, definition), instances in sorted(
        eligible.items(), key=lambda x: x[0][0]
    ):
        if len(instances) > instances_per_sense:
            idxs = rng.choice(len(instances), size=instances_per_sense, replace=False)
            instances = [instances[int(i)] for i in idxs]
        sampled.append((sense_uuid, form, definition, instances))

    if max_senses is not None:
        sampled = sampled[:max_senses]

    # Load docs needed for context extraction
    needed_doc_ids = list(
        {doc_id for _, _, _, insts in sampled for doc_id, _, _ in insts}
    )
    docs_df = (
        pl.scan_parquet(str(docs))
        .filter(pl.col("doc_id").is_in(needed_doc_ids))
        .collect(engine="streaming")
    )
    docs_map: dict[str, str] = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    batch_requests: list[str] = []
    metadata_rows: list[str] = []

    for i, (sense_uuid, form, definition, instances) in enumerate(sampled):
        # Build numbered context list; skip instances with missing docs
        contexts: list[str] = []
        valid_instances: list[tuple[str, int, str]] = []
        for doc_id, byte_offset, surface_form in instances:
            text = docs_map.get(doc_id, "")
            if not text:
                continue
            ctx = extract_context(text, byte_offset, surface_form, context_chars)
            contexts.append(ctx)
            valid_instances.append((doc_id, byte_offset, surface_form))

        if len(valid_instances) < min_instances:
            continue

        system_msg = build_critic_system_message(form, definition)
        numbered = "\n".join(
            f'{j + 1}. "...{ctx}..."' for j, ctx in enumerate(contexts)
        )
        user_msg = (
            f"{numbered}\n\n"
            f'Respond ONLY with valid JSON: {{"bad_indices": [1, 3]}} listing numbers '
            f"that do NOT match the sense. Use [] if all match."
        )

        custom_id = str(i)
        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 300,
        }
        if model in _JSON_MODE_MODELS:
            body["response_format"] = {"type": "json_object"}
        batch_requests.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        )
        metadata_rows.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "form": form,
                    "sense_uuid": sense_uuid,
                    "sense_definition": definition,
                    "model": model,
                    "instances": [
                        {"doc_id": doc_id, "byte_offset": byte_offset}
                        for doc_id, byte_offset, _ in valid_instances
                    ],
                }
            )
        )

    print(f"Prepared {len(batch_requests)} critic requests")
    if not batch_requests:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[Path, Path]] = []
    total = len(batch_requests)
    n_chunks = max(1, (total + max_batch_size - 1) // max_batch_size)
    for chunk_idx in range(n_chunks):
        start = chunk_idx * max_batch_size
        end = min(start + max_batch_size, total)
        chunk_num = chunk_idx + 1
        batch_path = out_dir / f"critic_input_{batch_id}_{chunk_num:03d}.jsonl"
        metadata_path = out_dir / f"critic_metadata_{batch_id}_{chunk_num:03d}.jsonl"
        with batch_path.open("w") as f:
            for line in batch_requests[start:end]:
                f.write(line + "\n")
        with metadata_path.open("w") as f:
            for line in metadata_rows[start:end]:
                f.write(line + "\n")
        print(
            f"Chunk {chunk_num}/{n_chunks}: {end - start} requests → {batch_path.name}"
        )
        chunks.append((batch_path, metadata_path))

    print(f"Wrote {total} total requests in {n_chunks} chunk(s) to {out_dir}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Groq critic batch JSONL for sense-label validation"
    )
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--instances-per-sense",
        type=int,
        default=20,
        help="Max instances to include per sense (default: 20)",
    )
    parser.add_argument(
        "--min-instances",
        type=int,
        default=5,
        help="Skip senses with fewer instances needing review (default: 5)",
    )
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=50_000)
    parser.add_argument(
        "--max-senses",
        type=int,
        default=None,
        help="Cap the number of senses included in the batch (default: no limit)",
    )
    args = parser.parse_args()

    run(
        senses_db=args.senses_db,
        labeled_db=args.labeled_db,
        docs=args.docs,
        output_dir=args.output_dir,
        instances_per_sense=args.instances_per_sense,
        min_instances=args.min_instances,
        model=args.model,
        context_chars=args.context_chars,
        seed=args.seed,
        max_batch_size=args.max_batch_size,
        max_senses=args.max_senses,
    )


if __name__ == "__main__":
    main()
