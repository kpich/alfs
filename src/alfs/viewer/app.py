"""Flask viewer for ALFS dictionary data.

Usage:
    python -m alfs.viewer.app
"""

import json
import math
from pathlib import Path

from flask import Flask, abort, render_template, request

DATA_PATH = Path("../viewer_data/data.json")
QC_STATS_PATH = Path("../viewer_data/qc_stats.json")
QC_LAG_PATH = Path("../viewer_data/qc_lag.json")
QC_COVERAGE_PATH = Path("../viewer_data/qc_coverage.json")

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
    total = len(data["entries"])
    return render_template("index.html", total=total, query=None, results=None)


@app.route("/search")
def search():
    data = get_data()
    q = request.args.get("q", "").strip()
    total = len(data["entries"])
    results = None
    if q:
        results = sorted(
            [
                (form, entry)
                for form, entry in data["entries"].items()
                if q.lower() in form.lower()
            ],
            key=lambda x: x[0].lower(),
        )
    return render_template("index.html", total=total, query=q, results=results)


@app.route("/list")
def listing():
    data = get_data()
    page = request.args.get("page", 1, type=int)
    recent = _recent_forms(data)
    sorted_entries = sorted(data["entries"].items(), key=lambda x: x[0].lower())
    total = len(sorted_entries)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    return render_template(
        "list.html",
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

    # Detect parent (morph-base) form and load its entry
    base_forms = {s["morph_base"] for s in senses if s.get("morph_base")}
    parent_form = base_forms.pop() if len(base_forms) == 1 else None
    parent_entry = data["entries"].get(parent_form) if parent_form else None
    parent_senses = parent_entry["senses"] if parent_entry else None

    has_chart = bool(by_year_kde) or bool(
        parent_entry and parent_entry.get("by_year_kde")
    )
    chart_data = {}
    if has_chart:
        traces = []
        for sk in [s["key"] for s in senses]:
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
        if parent_entry:
            parent_kde = parent_entry.get("by_year_kde", {})
            for sk in [s["key"] for s in parent_entry["senses"]]:
                pts = parent_kde.get(sk)
                if pts:
                    traces.append(
                        {
                            "type": "scatter",
                            "mode": "lines",
                            "name": f"{parent_form}:{sk}",
                            "line": {"dash": "dot"},
                            "x": [p[0] for p in pts],
                            "y": [p[1] for p in pts],
                        }
                    )
        chart_data = {
            "traces": traces,
            "layout": {
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "share of corpus", "tickformat": ".3%"},
                "width": 500,
                "height": 300,
                "autosize": False,
            },
        }

    senses_bar = entry.get("senses_bar", [])
    all_bar = list(senses_bar)
    if parent_entry:
        for sb in parent_entry.get("senses_bar", []):
            all_bar.append(
                {
                    "key": f"{parent_form}:{sb['key']}",
                    "pos": sb["pos"],
                    "proportion": sb["proportion"],
                }
            )
    has_bar_chart = len(all_bar) >= 2
    bar_chart_data: dict = {}
    if has_bar_chart:
        bar_traces = []
        for sb in all_bar:
            bar_traces.append(
                {
                    "type": "bar",
                    "name": sb["key"],
                    "x": [sb["pos"] or "untagged"],
                    "y": [sb["proportion"]],
                }
            )
        bar_chart_data = {
            "traces": bar_traces,
            "layout": {
                "barmode": "stack",
                "xaxis": {"title": "Part of speech"},
                "yaxis": {"title": "share of instances", "tickformat": ".0%"},
                "width": 500,
                "height": 300,
                "autosize": False,
            },
        }

    return render_template(
        "word.html",
        form=form,
        senses=senses,
        parent_form=parent_form,
        parent_senses=parent_senses,
        has_chart=has_chart,
        chart_data=json.dumps(chart_data),
        has_bar_chart=has_bar_chart,
        bar_chart_data=json.dumps(bar_chart_data),
        percentile=percentile,
        is_recent=is_recent,
    )


@app.route("/qc")
def qc():
    if not QC_STATS_PATH.exists():
        return render_template(
            "qc.html", available=False, stats=None, lag=None, coverage=None
        )
    stats = json.loads(QC_STATS_PATH.read_text())
    lag = json.loads(QC_LAG_PATH.read_text()) if QC_LAG_PATH.exists() else None
    coverage = (
        json.loads(QC_COVERAGE_PATH.read_text()) if QC_COVERAGE_PATH.exists() else None
    )
    return render_template(
        "qc.html", available=True, stats=stats, lag=lag, coverage=coverage
    )


@app.route("/qc/<int:rating>")
def qc_instances(rating: int):
    if rating not in (0, 1):
        abort(404)
    instances_path = Path(f"../viewer_data/qc_{rating}.json")
    if not instances_path.exists():
        return render_template(
            "qc_instances.html",
            available=False,
            rating=rating,
            instances=[],
            page=1,
            total_pages=1,
            total=0,
        )
    data = json.loads(instances_path.read_text())
    all_instances = data["instances"]
    total = len(all_instances)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = request.args.get("page", 1, type=int)
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    return render_template(
        "qc_instances.html",
        available=True,
        rating=rating,
        instances=all_instances[start : start + PAGE_SIZE],
        page=page,
        total_pages=total_pages,
        total=total,
    )


def main() -> None:
    app.run(host="localhost", port=5001, debug=False)


if __name__ == "__main__":
    main()
