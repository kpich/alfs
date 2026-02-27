"""Flask viewer for ALFS dictionary data.

Usage:
    python -m alfs.viewer.app
"""

import json
from pathlib import Path

from flask import Flask, abort, render_template_string

DATA_PATH = Path("viewer_data/data.json")

app = Flask(__name__)

_data: dict | None = None


def get_data() -> dict:
    global _data
    if _data is None:
        _data = json.loads(DATA_PATH.read_text())
    return _data


INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ALFS</title></head>
<body>
<h1>ALFS Word List</h1>
<ul>
{% for form in forms %}
  <li><a href="/word/{{ form }}">{{ form }}</a></li>
{% endfor %}
</ul>
</body>
</html>
"""

WORD_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ form }}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h1>{{ form }}</h1>

<h2>Senses</h2>
<ul>
{% for sense in senses %}
  <li><strong>{{ sense.key }}</strong> — {{ sense.definition }}
    {% if sense.subsenses %}
    <ul>
      {% for sub in sense.subsenses %}
      <li><strong>{{ sub.key }}</strong> — {{ sub.definition }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  </li>
{% endfor %}
</ul>

{% if has_chart %}
<h2>Usage by Year</h2>
<div id="chart"></div>
<script>
var chartData = {{ chart_data | safe }};
Plotly.newPlot('chart', chartData.traces, chartData.layout);
</script>
{% endif %}

<p><a href="/">← Back to word list</a></p>
</body>
</html>
"""


@app.route("/")
def index():
    data = get_data()
    forms = sorted(data["entries"].keys())
    return render_template_string(INDEX_TEMPLATE, forms=forms)


@app.route("/word/<form>")
def word(form: str):
    data = get_data()
    entry = data["entries"].get(form)
    if entry is None:
        abort(404)

    senses = entry["senses"]
    by_year = entry.get("by_year", {})

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
            },
        }

    return render_template_string(
        WORD_TEMPLATE,
        form=form,
        senses=senses,
        has_chart=has_chart,
        chart_data=json.dumps(chart_data),
    )


def main() -> None:
    app.run(host="localhost", port=5000, debug=False)


if __name__ == "__main__":
    main()
