from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template

from alfs.data_models.alf import Sense
from alfs.data_models.change_store import Change, ChangeStatus, ChangeStore, ChangeType
from alfs.data_models.sense_store import SenseStore

app = Flask(__name__)
_sense_store: SenseStore | None = None
_change_store: ChangeStore | None = None


def apply_change(change: Change, sense_store: SenseStore) -> None:
    if change.type == ChangeType.rewrite:
        after = [Sense.model_validate(s) for s in change.data["after"]]
        sense_store.update(
            change.form, lambda existing: existing.model_copy(update={"senses": after})
        )  # type: ignore[union-attr]


def _change_to_dict(change: Change) -> dict:  # type: ignore[type-arg]
    return {
        "id": change.id,
        "type": change.type.value,
        "form": change.form,
        "data": change.data,
        "status": change.status.value,
        "created_at": change.created_at.isoformat(),
        "reviewed_at": change.reviewed_at.isoformat() if change.reviewed_at else None,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/changes")
def list_changes():
    assert _change_store is not None
    return jsonify([_change_to_dict(c) for c in _change_store.all_pending()])


@app.post("/api/changes/<id>/approve")
def approve_change(id: str):
    assert _change_store is not None
    assert _sense_store is not None
    change = _change_store.get(id)
    if change is None:
        return jsonify({"error": "not found"}), 404
    if change.status != ChangeStatus.pending:
        return jsonify({"error": "change is not pending"}), 409
    apply_change(change, _sense_store)
    _change_store.set_status(id, ChangeStatus.approved, reviewed_at=datetime.utcnow())
    return jsonify({"ok": True})


@app.post("/api/changes/<id>/reject")
def reject_change(id: str):
    assert _change_store is not None
    change = _change_store.get(id)
    if change is None:
        return jsonify({"error": "not found"}), 404
    if change.status != ChangeStatus.pending:
        return jsonify({"error": "change is not pending"}), 409
    _change_store.set_status(id, ChangeStatus.rejected, reviewed_at=datetime.utcnow())
    return jsonify({"ok": True})


def main(senses_db: Path, changes_db: Path, port: int = 5003) -> None:
    global _sense_store, _change_store
    _sense_store = SenseStore(senses_db)
    _change_store = ChangeStore(changes_db)
    app.run(host="127.0.0.1", port=port, debug=False)
