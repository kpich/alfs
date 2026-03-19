"""Incrementally augment the docs corpus from a MediaWiki dump.

Usage:
    python -m alfs.etl.augment \
        --source wikibooks \
        --corpus ../text_data/docs.parquet \
        --cache-dir ../text_data/cache \
        --ngram-cache ../text_data/ngram_cache.npy \
        [--n-docs 10000]

Each source tracks a cursor in {cache_dir}/{source}_cursor.json so successive
runs pick up from where the previous run stopped.  Run until "0 new docs" is
printed to signal that the dump has been fully consumed.
"""

import argparse
from collections.abc import Iterator
import json
from pathlib import Path

from alfs.etl.corpus import append_docs, get_doc_ids
from alfs.etl.ngram_cache import NgramCache
from alfs.etl.parse_dump import parse_page
from alfs.etl.sources import SOURCES, Source
from alfs.etl.stream_dump import stream_pages


def get_streamer(source: Source, dump_path: Path) -> Iterator[dict]:
    if source.type == "mediawiki":
        return stream_pages(dump_path, source.name)
    elif source.type == "gutenberg":
        from alfs.etl.stream_gutenberg import stream_gutenberg

        return stream_gutenberg(dump_path)
    elif source.type == "hf":
        from alfs.etl.stream_hf import stream_hf

        return stream_hf(source.hf_dataset)
    else:
        raise ValueError(f"Unknown source type: {source.type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally augment corpus from a MediaWiki dump"
    )
    parser.add_argument(
        "--source", required=True, choices=list(SOURCES), help="Source name"
    )
    parser.add_argument("--corpus", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--cache-dir", required=True, help="Directory containing cached dump file"
    )
    parser.add_argument("--ngram-cache", required=True, help="Path to ngram_cache.npy")
    parser.add_argument(
        "--n-docs", type=int, default=10000, help="Target number of new docs to add"
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=200,
        help="Skip docs with fewer than this many chars of text (filters stubs)",
    )
    args = parser.parse_args()

    source_name = args.source
    corpus_path = Path(args.corpus)
    cache_dir = Path(args.cache_dir)
    ngram_cache_path = Path(args.ngram_cache)

    source = SOURCES[source_name]
    dump_path = cache_dir / source.dump_filename

    if source.type != "hf" and not dump_path.exists():
        raise FileNotFoundError(
            f"Dump not found: {dump_path}. Run: python -m alfs.etl.download "
            f"--source {source_name} --cache-dir {cache_dir}"
        )

    # 1. Load existing doc_ids
    existing_ids: set[str] = set()
    if corpus_path.exists():
        existing_ids = get_doc_ids(corpus_path)
        print(f"Corpus has {len(existing_ids)} existing docs")
    else:
        print("No existing corpus — starting fresh")

    # 2. Load ngram cache
    if ngram_cache_path.exists():
        cache = NgramCache.load(ngram_cache_path)
        print(f"Loaded ngram cache ({len(cache._hashes):,} hashes)")
    else:
        cache = NgramCache()
        print("Starting with empty ngram cache")

    # 3. Read cursor
    cursor_path = cache_dir / f"{source_name}_cursor.json"
    pages_consumed = 0
    if cursor_path.exists():
        with open(cursor_path) as f:
            pages_consumed = json.load(f)["pages_consumed"]
        print(f"Cursor: resuming after {pages_consumed:,} pages")
    else:
        print("No cursor found — starting from beginning of dump")

    # 4-6. Stream, skip, parse, dedup, collect
    new_docs = []
    pages_skipped = 0
    pages_processed = 0
    exact_dupes = 0
    ngram_dupes = 0

    for page in get_streamer(source, dump_path):
        if pages_skipped < pages_consumed:
            pages_skipped += 1
            continue

        pages_processed += 1

        doc = parse_page(page, source_name)

        if len(doc.text) < args.min_text_len:
            continue

        # exact dedup
        if doc.doc_id in existing_ids:
            exact_dupes += 1
            continue
        existing_ids.add(doc.doc_id)

        # ngram near-dedup
        if cache.is_near_duplicate(doc.text):
            ngram_dupes += 1
            continue
        cache.add_doc(doc.text)

        new_docs.append(doc)
        if len(new_docs) >= args.n_docs:
            break

    print(
        f"Scanned {pages_processed} pages: "
        f"{len(new_docs)} new, {exact_dupes} exact dupes, {ngram_dupes} near dupes"
    )

    # 7. Save updated cursor
    new_pages_consumed = pages_consumed + pages_processed
    with open(cursor_path, "w") as f:
        json.dump({"pages_consumed": new_pages_consumed}, f)
    print(f"Cursor saved: {new_pages_consumed:,} total pages consumed")

    # 8. Append docs + save ngram cache
    if new_docs:
        append_docs(new_docs, corpus_path)
        print(f"Appended {len(new_docs)} docs to {corpus_path}")
    else:
        print("0 new docs — dump may be fully consumed")

    cache.save(ngram_cache_path)
    print(f"Ngram cache saved to {ngram_cache_path}")


if __name__ == "__main__":
    main()
