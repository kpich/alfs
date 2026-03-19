"""Download and cache a corpus dump for a given source.

Usage:
    python -m alfs.etl.download --source wikibooks --cache-dir ../text_data/cache
"""

import argparse
from pathlib import Path
import urllib.request

from alfs.etl.sources import SOURCES


def download(source_name: str, cache_dir: Path) -> Path | None:
    """Download dump if not already cached; return path to local file, or None for HF
    sources."""
    source = SOURCES[source_name]

    if source.type == "hf":
        print(
            f"No download needed for HuggingFace source '{source_name}' — "
            f"data streams on demand."
        )
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    dump_path = cache_dir / source.dump_filename

    if dump_path.exists():
        print(f"Dump already cached at {dump_path}")
        return dump_path

    print(f"Downloading {source.dump_url} ...")

    def _progress(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(source.dump_url, dump_path, _progress)
    print(f"\nSaved to {dump_path}")
    return dump_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a corpus dump")
    parser.add_argument(
        "--source", required=True, choices=list(SOURCES), help="Source name"
    )
    parser.add_argument(
        "--cache-dir", required=True, help="Directory to cache downloaded dumps"
    )
    args = parser.parse_args()
    download(args.source, Path(args.cache_dir))


if __name__ == "__main__":
    main()
