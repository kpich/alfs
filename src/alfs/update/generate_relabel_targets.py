"""Generate one UpdateTarget JSON file per entry in alfs.json.

Usage:
    python -m alfs.update.generate_relabel_targets \\
        --alfs alfs.json --output-dir targets/
"""

import argparse
from pathlib import Path
from urllib.parse import quote

from alfs.data_models.alf import Alfs
from alfs.data_models.update_target import UpdateTarget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one target JSON file per form in alfs.json"
    )
    parser.add_argument("--alfs", required=True, help="Path to alfs.json")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for target JSON files"
    )
    args = parser.parse_args()

    alfs = Alfs.model_validate_json(Path(args.alfs).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for form in alfs.entries:
        target = UpdateTarget(form=form)
        safe = quote(form, safe="")
        (out_dir / f"{safe}.json").write_text(target.model_dump_json())
        print(f"  {form}")

    print(f"Wrote {len(alfs.entries)} targets to {out_dir}")


if __name__ == "__main__":
    main()
