"""Merge partial compile outputs into a single data.json with percentile ranks.

Usage:
    python -m alfs.viewer.compile_merge \\
        --inputs entries_0.json entries_1.json ... \\
        --corpus-counts corpus_counts.json \\
        --output data.json
"""

import argparse
import json
from pathlib import Path

from alfs.viewer.compile import assign_percentiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge batch compile outputs")
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="Partial entries JSON files"
    )
    parser.add_argument("--corpus-counts", required=True, help="corpus_counts.json")
    parser.add_argument("--output", required=True, help="Output data.json path")
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    for path in args.inputs:
        partial = json.loads(Path(path).read_text())
        merged.update(partial["entries"])

    corpus_counts: dict[str, int] = json.loads(Path(args.corpus_counts).read_text())
    assign_percentiles(merged, corpus_counts)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"entries": merged}, indent=2))
    print(f"Merged {len(args.inputs)} batches → {len(merged)} entries → {args.output}")


if __name__ == "__main__":
    main()
