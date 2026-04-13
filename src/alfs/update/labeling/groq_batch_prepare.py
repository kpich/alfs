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
import math
from pathlib import Path

import numpy as np
import polars as pl

from alfs.data_models.alf import Alf, morph_base_form
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.seg.aggregate_occurrences import prefix as form_prefix
from alfs.update.labeling.label_occurrences import build_sense_menu, extract_context


def compute_sense_quality_counts(
    occ_store: OccurrenceStore,
) -> dict[str, dict[str, int]]:
    """Return {form: {sense_uuid: count_of_rating=2}} from labeled occurrences."""
    df = occ_store.to_polars()
    if len(df) == 0:
        return {}
    result: dict[str, dict[str, int]] = defaultdict(dict)
    counts = (
        df.filter(pl.col("rating") == 2)
        .group_by(["form", "sense_key"])
        .agg(pl.len().alias("count"))
    )
    for row in counts.iter_rows(named=True):
        result[str(row["form"])][str(row["sense_key"])] = int(row["count"])
    return result


def sense_weight(
    alf: Alf,
    store: SenseStore,
    quality_counts: dict[str, dict[str, int]],
) -> float:
    """Σ_i 1/sqrt(N_i + 1) across all senses visible in this form's menu.

    N_i is the count of rating=2 labels for sense i. Mirrors the case-variant
    and morph_base traversal in build_sense_menu.
    """
    variants = store.read_case_variants(alf.form)
    weight = 0.0
    for variant in variants:
        form_counts = quality_counts.get(variant.form, {})
        for sense in variant.senses:
            n_i = form_counts.get(sense.id, 0)
            weight += 1.0 / math.sqrt(n_i + 1)
        base_name = morph_base_form(variant)
        if base_name is not None:
            base_alf = store.read(base_name)
            if base_alf is not None:
                base_counts = quality_counts.get(base_name, {})
                for sense in base_alf.senses:
                    n_i = base_counts.get(sense.id, 0)
                    weight += 1.0 / math.sqrt(n_i + 1)
    return weight


def allocate_proportional(
    sense_weights: dict[str, float],
    good_labeled: dict[str, int],
    corpus_counts: dict[str, int],
    budget: int,
    min_count: int = 5,
) -> dict[str, int]:
    """Allocate budget proportionally to sense_weight, capped by available instances.

    available[f] = corpus[f] - good_labeled[f] (unlabeled + rating-0 instances).
    Uses iterative proportional capping: forms whose proportional share exceeds
    their available pool are given their max and removed; the remaining budget is
    redistributed among the rest.
    """
    eligible = {
        f: w
        for f, w in sense_weights.items()
        if corpus_counts.get(f, 0) >= min_count and w > 0
    }

    def avail(f: str) -> int:
        return max(0, corpus_counts.get(f, 0) - good_labeled.get(f, 0))

    alloc: dict[str, float] = {}
    active = {f: w for f, w in eligible.items() if avail(f) > 0}
    remaining = float(budget)

    while active and remaining > 0:
        total_w = sum(active.values())
        shares = {f: remaining * w / total_w for f, w in active.items()}
        capped = {f for f, s in shares.items() if s >= avail(f)}
        if not capped:
            alloc.update(shares)
            break
        for f in capped:
            alloc[f] = float(avail(f))
            remaining -= avail(f)
        active = {f: w for f, w in active.items() if f not in capped}

    return {f: int(a) for f, a in alloc.items() if int(a) > 0}


def split_labeled_pairs(
    good_labeled_df: pl.DataFrame,
    max_sense_ts: dict[str, str],
) -> tuple[dict[str, set[tuple[str, int]]], dict[str, list[tuple[str, int]]]]:
    """Classify rating >= 1 labeled occurrences into fresh-good and stale sets.

    Returns (good_pairs, stale_pairs) where:
      good_pairs  — labeled AFTER (or same time as) the latest sense addition;
                    excluded from sampling
      stale_pairs — labeled BEFORE the latest sense was added; eligible for
                    re-sampling
    """
    good_pairs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    stale_pairs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in good_labeled_df.iter_rows(named=True):
        form = str(row["form"])
        pair = (str(row["doc_id"]), int(row["byte_offset"]))
        form_max_ts = max_sense_ts.get(form)
        labeled_ts = row["updated_at"]
        if form_max_ts and labeled_ts and labeled_ts < form_max_ts:
            stale_pairs[form].append(pair)
        else:
            good_pairs[form].add(pair)
    return good_pairs, stale_pairs


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
        f"Rating: 2=excellent match for this sense, 1=okay but could be more specific,"
        f" 0=the word is NOT being used in any of these senses (e.g., it's a different"
        f" meaning entirely, it's being used as a proper name, it's slang for something"
        f" else). When in doubt between 0 and 1, use 0.\n"
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
    stale_fraction: float = 0.1,
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
    quality_counts = compute_sense_quality_counts(occ_store)

    # Load corpus counts
    parquet_files = list(Path(seg_data_dir).glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No occurrences.parquet files found in {seg_data_dir}")
    corpus_total: pl.DataFrame = (
        pl.concat(
            [
                pl.scan_parquet(str(f)).select(
                    pl.col("form").str.to_lowercase().alias("form")
                )
                for f in parquet_files
            ]
        )
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

    # Load good_labeled counts (rating >= 1) to determine available instances.
    labeled_df = occ_store.count_by_form()
    good_labeled: dict[
        str, int
    ] = {}  # rating >= 1 (these instances won't be resampled)
    for row in labeled_df.iter_rows(named=True):
        f = row["form"]
        good_labeled[f] = row["n_total"] - row["n_bad"]

    # Compute per-form sense weights (case variants share an occurrence pool via
    # case-insensitive lookup, so deduplicate to one representative per lowercase
    # cluster to avoid sampling the same pool twice).
    all_entries = sense_store.all_entries()
    form_weights: dict[str, float] = {}
    seen_lower: set[str] = set()
    for form, alf in all_entries.items():
        if form.lower() in seen_lower:
            continue
        seen_lower.add(form.lower())
        w = sense_weight(alf, sense_store, quality_counts)
        if w > 0:
            form_weights[form] = w

    # Use corpus counts keyed by lowercase form (parquets are lowercased).
    agg_corpus = {f: corpus_counts.get(f.lower(), 0) for f in form_weights}

    # Allocate budget proportionally to sense weight
    allocation = allocate_proportional(
        sense_weights=form_weights,
        good_labeled=good_labeled,
        corpus_counts=agg_corpus,
        budget=n,
        min_count=min_count,
    )
    print(
        f"Allocating {sum(allocation.values())} instances "
        f"across {len(allocation)} forms"
    )

    # Build pair sets for candidate filtering:
    #   good_pairs  — rating >= 1, labeled AFTER latest sense was added
    #                 → excluded from sampling
    #   stale_pairs — rating >= 1, labeled BEFORE latest sense was added
    #                 → eligible for re-sampling
    max_sense_ts = sense_store.max_sense_updated_at_by_form()
    all_labeled = occ_store.to_polars()
    good_pairs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    stale_pairs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    if len(all_labeled) > 0:
        good_labeled_df = all_labeled.filter(pl.col("rating").is_in([1, 2])).select(
            ["form", "doc_id", "byte_offset", "updated_at"]
        )
        good_pairs, stale_pairs = split_labeled_pairs(good_labeled_df, max_sense_ts)
    else:
        good_pairs = defaultdict(set)
        stale_pairs = defaultdict(list)

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
        lookup_forms = list({f.lower() for f in forms})
        df = pl.read_parquet(str(occ_path)).filter(
            pl.col("form").str.to_lowercase().is_in(lookup_forms)
        )
        for form in forms:
            k = allocation[form]
            if k <= 0:
                continue
            form_df = df.filter(pl.col("form").str.to_lowercase() == form.lower())

            # Stale pairs for this form (labeled.db stores under canonical entry form)
            form_stale: list[tuple[str, int]] = stale_pairs.get(form, [])
            form_stale_set: set[tuple[str, int]] = set(form_stale)

            fresh_candidates: list[tuple[str, int]] = [
                (str(row["doc_id"]), int(row["byte_offset"]))
                for row in form_df.select(["doc_id", "byte_offset"]).iter_rows(
                    named=True
                )
                if (str(row["doc_id"]), int(row["byte_offset"]))
                not in good_pairs.get(form, set())
                and (str(row["doc_id"]), int(row["byte_offset"])) not in form_stale_set
            ]

            k_stale = min(int(k * stale_fraction), len(form_stale))
            k_fresh = k - k_stale

            if form_stale and k_stale > 0:
                chosen_stale = rng.choice(len(form_stale), size=k_stale, replace=False)
                for idx in chosen_stale:
                    doc_id, byte_offset = form_stale[int(idx)]
                    sampled.append(
                        {"form": form, "doc_id": doc_id, "byte_offset": byte_offset}
                    )

            if fresh_candidates and k_fresh > 0:
                chosen_fresh = rng.choice(
                    len(fresh_candidates),
                    size=min(k_fresh, len(fresh_candidates)),
                    replace=False,
                )
                for idx in chosen_fresh:
                    doc_id, byte_offset = fresh_candidates[int(idx)]
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
    parser.add_argument(
        "--stale-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of each form's budget reserved for re-labeling stale "
            "occurrences (labeled before the most recently added sense). "
            "Default: 0.1"
        ),
    )
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
        stale_fraction=args.stale_fraction,
    )


if __name__ == "__main__":
    main()
