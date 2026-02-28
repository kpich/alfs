from pathlib import Path
import threading
import time
import webbrowser

from .app import main

project_root = Path(__file__).resolve().parent.parent.parent.parent

PORT = 5002


def _open_browser() -> None:
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")


t = threading.Thread(target=_open_browser, daemon=True)
t.start()

main(project_root, port=PORT)
