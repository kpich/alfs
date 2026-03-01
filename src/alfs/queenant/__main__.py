import argparse
from pathlib import Path
import threading
import time
import webbrowser

from alfs.queenant import app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queenant: human approval UI for sense changes"
    )
    parser.add_argument("--senses-db", default="../alfs_data/senses.db")
    parser.add_argument("--changes-db", default="../alfs_data/changes.db")
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()

    port = args.port

    def _open() -> None:
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_open, daemon=True).start()
    app.main(Path(args.senses_db), Path(args.changes_db), port=port)


main()
