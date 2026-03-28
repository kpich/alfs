"""Prepare a Groq batch file for sense labeling.

Usage:
    python -m alfs.update.labeling.groq_batch_prepare \\
        --senses-db senses.db --labeled-db labeled.db \\
        --seg-data-dir by_prefix/ --docs docs.parquet \\
        --output-dir groq_batch/ [--n 100000] [--model llama-3.1-8b-instant] \\
        [--context-chars 150] [--seed 42]

Output files in --output-dir:
    batch_input.jsonl   Groq batch API input; upload this to Groq.
    batch_metadata.jsonl  Sidecar: custom_id -> {form, doc_id, byte_offset}.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import polars as pl

from alfs.data_models.alf import Alf, morph_base_form
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.seg.aggregate_occurrences import prefix as form_prefix
from alfs.update.labeling.label_occurrences import build_sense_menu, extract_context


def effective_sense_count(alf: Alf, store: SenseStore) -> int:
    """Count of numbered options in the sense menu when labeling this form.

    Includes the form's own senses plus, if all senses share a morph_base,
    the base form's senses (mirrors build_sense_menu behaviour).
    """
    target = alf
    if alf.redirect is not None:
        redirect_alf = store.read(alf.redirect)
        if redirect_alf is None:
            return 0
        target = redirect_alf

    count = len(target.senses)
    base_name = morph_base_form(target)
    if base_name is not None:
        base_alf = store.read(base_name)
        if base_alf is not None:
            count += len(base_alf.senses)
    return count


def allocate_instances(
    effective_senses: dict[str, int],
    existing_labeled: dict[str, int],
    good_labeled: dict[str, int],
    corpus_counts: dict[str, int],
    budget: int,
    min_count: int = 5,
) -> dict[str, int]:
    """Allocate labeling budget across forms for equal expected coverage per sense.

    Finds k (target labels per sense) via binary search such that
        sum_f clip(k * senses[f] - existing[f], 0, available[f]) ≈ budget
    where available[f] = corpus[f] - good_labeled[f] (unlabeled + rating-0 instances).

    existing_labeled is used for allocation weight; good_labeled (rating >= 1) is
    used to determine which instances are already well-labeled and should not be
    re-sampled.
    """
    eligible = [
        f
        for f in effective_senses
        if corpus_counts.get(f, 0) >= min_count and effective_senses[f] > 0
    ]

    def avail(f: str) -> int:
        return max(0, corpus_counts.get(f, 0) - good_labeled.get(f, 0))

    def need_at_k(f: str, k: float) -> float:
        return min(
            max(0.0, k * effective_senses[f] - existing_labeled.get(f, 0)),
            float(avail(f)),
        )

    def total_at_k(k: float) -> float:
        return sum(need_at_k(f, k) for f in eligible)

    max_total = sum(avail(f) for f in eligible)
    if max_total <= budget:
        return {f: avail(f) for f in eligible if avail(f) > 0}

    # Find hi such that total_at_k(hi) >= budget.
    hi = float(max(budget, 1))
    while total_at_k(hi) < budget:
        hi *= 2.0

    lo = 0.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if total_at_k(mid) < budget:
            lo = mid
        else:
            hi = mid

    k = lo
    return {f: int(need_at_k(f, k)) for f in eligible if int(need_at_k(f, k)) > 0}


def build_system_message(form: str, sense_menu: str) -> str:
    """System message for the labeling prompt (same per form, cached by Groq)."""
    return (
        f"You are a word sense tagger for English.\n"
        f"\n"
        f'Given a sentence containing "{form}", identify which numbered sense applies,'
        f" rate the fit, and list substitute words.\n"
        f"\n"
        f'Respond with ONLY valid JSON: {{"sense_key": "1", "rating": 2,'
        f' "synonyms": ["word1", "word2"]}}\n'
        f"Rating: 2=excellent, 1=okay (needs a more refined sense), 0=poor/doesn't "
        f"fit.\n"
        f'If rating is 0, set sense_key to "0".\n'
        f'synonyms: other words that could roughly fit in place of "{form}" here.'
        f" Doesn't need to be a perfect match — approximate or related words are fine."
        f" Use [] if nothing fits at all.\n"
        f"\n"
        f'Senses of "{form}":\n'
        f"{sense_menu}"
    )


def run(
    senses_db: str | Path,
    labeled_db: str | Path,
    seg_data_dir: str | Path,
    docs: str | Path,
    output_dir: str | Path,
    n: int = 100_000,
    model: str = "llama-3.1-8b-instant",
    context_chars: int = 100,
    seed: int | None = None,
    min_count: int = 5,
    max_batch_size: int = 50_000,
    batch_id: str | None = None,
) -> list[tuple[Path, Path]]:
    """Build batch_input and batch_metadata JSONL files in output_dir.

    Files are named batch_input_{batch_id}_{NNN:03d}.jsonl where batch_id
    defaults to a timestamp (YYYYMMDDTHHMMSS). If requests exceed max_batch_size,
    multiple chunk pairs are created.

    Returns list of (batch_path, metadata_path) pairs, one per chunk.
    """
    if batch_id is None:
        batch_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    sense_store = SenseStore(Path(senses_db))
    occ_store = OccurrenceStore(Path(labeled_db))

    # Load corpus counts
    parquet_files = list(Path(seg_data_dir).glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No occurrences.parquet files found in {seg_data_dir}")
    corpus_total: pl.DataFrame = (
        pl.concat([pl.scan_parquet(str(f)).select("form") for f in parquet_files])
        .group_by("form")
        .agg(pl.len().alias("total"))
        .collect()
    )
    corpus_counts: dict[str, int] = dict(
        zip(
            corpus_total["form"].to_list(),
            corpus_total["total"].to_list(),
            strict=False,
        )
    )

    # Load existing labeled counts (n_total for allocation weight; n_good for available)
    labeled_df = occ_store.count_by_form()
    existing_labeled: dict[str, int] = {}
    good_labeled: dict[
        str, int
    ] = {}  # rating >= 1 (these instances won't be resampled)
    for row in labeled_df.iter_rows(named=True):
        f = row["form"]
        existing_labeled[f] = row["n_total"]
        good_labeled[f] = row["n_total"] - row["n_bad"]

    # Compute effective sense counts; skip pure redirects
    all_entries = sense_store.all_entries()
    eff_senses: dict[str, int] = {}
    for form, alf in all_entries.items():
        ec = effective_sense_count(alf, sense_store)
        if ec > 0:
            eff_senses[form] = ec

    # Allocate budget
    allocation = allocate_instances(
        effective_senses=eff_senses,
        existing_labeled=existing_labeled,
        good_labeled=good_labeled,
        corpus_counts=corpus_counts,
        budget=n,
        min_count=min_count,
    )
    print(
        f"Allocating {sum(allocation.values())} instances "
        f"across {len(allocation)} forms"
    )

    # Build set of already-well-labeled (rating >= 1) pairs per form for filtering
    good_pairs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    all_labeled = occ_store.to_polars()
    if len(all_labeled) > 0:
        good_labeled_df = all_labeled.filter(pl.col("rating").is_in([1, 2])).select(
            ["form", "doc_id", "byte_offset"]
        )
        for row in good_labeled_df.iter_rows(named=True):
            good_pairs[str(row["form"])].add(
                (str(row["doc_id"]), int(row["byte_offset"]))
            )

    # Sample instances per form, grouped by prefix for efficient parquet loading
    rng = np.random.default_rng(seed)
    sampled: list[dict[str, object]] = []

    forms_by_prefix: dict[str, list[str]] = defaultdict(list)
    for form in allocation:
        forms_by_prefix[form_prefix(form)].append(form)

    for prefix_key, forms in sorted(forms_by_prefix.items()):
        occ_path = Path(seg_data_dir) / prefix_key / "occurrences.parquet"
        if not occ_path.exists():
            continue
        df = pl.read_parquet(str(occ_path)).filter(pl.col("form").is_in(forms))
        for form in forms:
            k = allocation[form]
            if k <= 0:
                continue
            form_df = df.filter(pl.col("form") == form)
            excluded = good_pairs.get(form, set())
            candidates: list[tuple[str, int]] = [
                (str(row["doc_id"]), int(row["byte_offset"]))
                for row in form_df.select(["doc_id", "byte_offset"]).iter_rows(
                    named=True
                )
                if (str(row["doc_id"]), int(row["byte_offset"])) not in excluded
            ]
            if not candidates:
                continue
            chosen = rng.choice(
                len(candidates), size=min(k, len(candidates)), replace=False
            )
            for idx in chosen:
                doc_id, byte_offset = candidates[int(idx)]
                sampled.append(
                    {"form": form, "doc_id": doc_id, "byte_offset": byte_offset}
                )

    print(f"Sampled {len(sampled)} instances")

    # Load docs needed for context extraction
    needed_doc_ids = list({str(s["doc_id"]) for s in sampled})
    docs_df = (
        pl.scan_parquet(str(docs))
        .filter(pl.col("doc_id").is_in(needed_doc_ids))
        .collect(engine="streaming")
    )
    docs_map: dict[str, str] = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    # Sort by form so same-word requests are consecutive (maximises prompt caching)
    sampled.sort(key=lambda x: str(x["form"]))

    sense_menu_cache: dict[str, tuple[str, dict[str, str]]] = {}
    batch_requests: list[str] = []
    metadata_rows: list[str] = []

    for i, item in enumerate(sampled):
        form = str(item["form"])
        doc_id = str(item["doc_id"])
        byte_offset = int(item["byte_offset"])  # type: ignore[call-overload]

        text = docs_map.get(doc_id, "")
        if not text:
            continue

        if form not in sense_menu_cache:
            try:
                menu, key_map = build_sense_menu(sense_store, form)
            except ValueError:
                continue
            sense_menu_cache[form] = (menu, key_map)
        sense_menu, key_map = sense_menu_cache[form]

        context = extract_context(text, byte_offset, form, context_chars)
        system_msg = build_system_message(form, sense_menu)
        user_msg = (
            f'The word "{form}" appears here: "...{context}..."\n\nWhich sense applies?'
        )

        custom_id = str(i)
        batch_requests.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "response_format": {"type": "json_object"},
                        "max_tokens": 150,
                    },
                }
            )
        )
        metadata_rows.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "form": form,
                    "doc_id": doc_id,
                    "byte_offset": byte_offset,
                    "model": model,
                    "key_map": key_map,
                }
            )
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split into chunks of max_batch_size
    chunks: list[tuple[Path, Path]] = []
    total = len(batch_requests)
    n_chunks = max(1, (total + max_batch_size - 1) // max_batch_size)
    for chunk_idx in range(n_chunks):
        start = chunk_idx * max_batch_size
        end = min(start + max_batch_size, total)
        chunk_requests = batch_requests[start:end]
        chunk_metadata = metadata_rows[start:end]
        chunk_num = chunk_idx + 1
        batch_path = out_dir / f"batch_input_{batch_id}_{chunk_num:03d}.jsonl"
        metadata_path = out_dir / f"batch_metadata_{batch_id}_{chunk_num:03d}.jsonl"
        with batch_path.open("w") as f:
            for line in chunk_requests:
                f.write(line + "\n")
        with metadata_path.open("w") as f:
            for line in chunk_metadata:
                f.write(line + "\n")
        print(
            f"Chunk {chunk_num}/{n_chunks}: {len(chunk_requests)} requests"
            f" → {batch_path.name}"
        )
        chunks.append((batch_path, metadata_path))

    print(f"Wrote {total} total requests in {n_chunks} chunk(s) to {out_dir}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Groq batch JSONL for sense labeling"
    )
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--seg-data-dir", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--context-chars", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-batch-size", type=int, default=50_000)
    args = parser.parse_args()

    run(
        senses_db=args.senses_db,
        labeled_db=args.labeled_db,
        seg_data_dir=args.seg_data_dir,
        docs=args.docs,
        output_dir=args.output_dir,
        n=args.n,
        model=args.model,
        context_chars=args.context_chars,
        seed=args.seed,
        min_count=args.min_count,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
