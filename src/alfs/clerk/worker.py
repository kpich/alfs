"""Entry point for the clerk queue worker.

Usage:
    python -m alfs.clerk.worker \\
        --queue-dir ../clerk_queue \\
        --senses-db ../alfs_data/senses.db \\
        [--labeled-db ../alfs_data/labeled.db] \\
        [--instance-log ../alfs_data/instance_log] \\
        [--watch] [--workers 4] [--poll-interval 1.0]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alfs.clerk.queue import drain, watch
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clerk: process queued sense mutations"
    )
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--labeled-db", default=None, help="Path to labeled.db (optional)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll for new requests; default is drain-and-exit",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--instance-log",
        default=None,
        help="Directory for instance-tagging JSONL change log",
    )
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    log_file = queue_dir / "clerk.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    sense_store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db)) if args.labeled_db else None
    instance_log = Path(args.instance_log) if args.instance_log else None

    if args.watch:
        watch(
            queue_dir,
            sense_store,
            occ_store,
            workers=args.workers,
            poll_interval=args.poll_interval,
            log_dir=instance_log,
        )
    else:
        drain(
            queue_dir,
            sense_store,
            occ_store,
            workers=args.workers,
            log_dir=instance_log,
        )


if __name__ == "__main__":
    main()
