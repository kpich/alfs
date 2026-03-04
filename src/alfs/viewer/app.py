"""Flask viewer for ALFS dictionary data.

Usage:
    python -m alfs.viewer.app
"""

import json
import math
from pathlib import Path

from flask import Flask, abort, render_template, request

DATA_PATH = Path("../viewer_data/data.json")

PAGE_SIZE = 500
RECENT_N = 100

app = Flask(__name__)

_data: dict | None = None


def _recent_forms(data: dict) -> set[str]:
    by_recency = sorted(
        data["entries"].items(),
        key=lambda x: x[1].get("updated_at") or "",
        reverse=True,
    )
    return {form for form, _ in by_recency[:RECENT_N]}


def get_data() -> dict:
    global _data
    if _data is None:
        _data = json.loads(DATA_PATH.read_text())
    return _data


@app.route("/")
def index():
    data = get_data()
    page = request.args.get("page", 1, type=int)
    recent = _recent_forms(data)
    sorted_entries = sorted(data["entries"].items(), key=lambda x: x[0].lower())
    total = len(sorted_entries)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    return render_template(
        "index.html",
        entries=sorted_entries[start : start + PAGE_SIZE],
        recent_forms=recent,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@app.route("/word/<form>")
def word(form: str):
    data = get_data()
    entry = data["entries"].get(form)
    if entry is None:
        abort(404)
    is_recent = form in _recent_forms(data)

    senses = entry["senses"]
    by_year_kde = entry.get("by_year_kde", {})
    percentile = entry["percentile"]

    has_chart = bool(by_year_kde)
    chart_data = {}
    if has_chart:
        sense_keys = [s["key"] for s in senses]
        traces = []
        for sk in sense_keys:
            pts = by_year_kde.get(sk)
            if pts:
                traces.append(
                    {
                        "type": "scatter",
                        "mode": "lines",
                        "name": sk,
                        "x": [p[0] for p in pts],
                        "y": [p[1] for p in pts],
                    }
                )

        chart_data = {
            "traces": traces,
            "layout": {
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "share of labeled", "tickformat": ".0%"},
                "width": 500,
                "height": 300,
                "autosize": False,
            },
        }

    return render_template(
        "word.html",
        form=form,
        senses=senses,
        has_chart=has_chart,
        chart_data=json.dumps(chart_data),
        percentile=percentile,
        is_recent=is_recent,
    )


def main() -> None:
    app.run(host="localhost", port=5001, debug=False)


if __name__ == "__main__":
    main()
