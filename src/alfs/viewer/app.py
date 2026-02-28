"""Flask viewer for ALFS dictionary data.

Usage:
    python -m alfs.viewer.app
"""

import json
from pathlib import Path

from flask import Flask, abort, render_template

DATA_PATH = Path("../viewer_data/data.json")

app = Flask(__name__)

_data: dict | None = None


def get_data() -> dict:
    global _data
    if _data is None:
        _data = json.loads(DATA_PATH.read_text())
    return _data


@app.route("/")
def index():
    data = get_data()
    entries = sorted(data["entries"].items(), key=lambda x: x[0].lower())
    return render_template("index.html", entries=entries)


@app.route("/word/<form>")
def word(form: str):
    data = get_data()
    entry = data["entries"].get(form)
    if entry is None:
        abort(404)

    senses = entry["senses"]
    by_year = entry.get("by_year", {})
    percentile = entry["percentile"]

    has_chart = bool(by_year)
    chart_data = {}
    if has_chart:
        years = sorted(by_year.keys(), key=int)
        sense_keys = [s["key"] for s in senses]

        traces = []
        for sk in sense_keys:
            y_vals = [by_year.get(yr, {}).get(sk, 0) for yr in years]
            if any(v > 0 for v in y_vals):
                traces.append(
                    {
                        "type": "bar",
                        "name": sk,
                        "x": years,
                        "y": y_vals,
                    }
                )

        chart_data = {
            "traces": traces,
            "layout": {
                "barmode": "stack",
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "Occurrences"},
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
    )


def main() -> None:
    app.run(host="localhost", port=5001, debug=False)


if __name__ == "__main__":
    main()
