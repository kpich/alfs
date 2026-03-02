from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template
import polars as pl

from alfs.corpus import fetch_instances
from alfs.data_models.alf import Alf, Sense, sense_key
from alfs.data_models.change_store import Change, ChangeStatus, ChangeStore, ChangeType
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore

app = Flask(__name__)
_sense_store: SenseStore | None = None
_change_store: ChangeStore | None = None
_occ_store: OccurrenceStore | None = None
_labeled: pl.DataFrame | None = None
_docs: pl.DataFrame | None = None


def apply_change(
    change: Change, sense_store: SenseStore, occ_store: OccurrenceStore | None = None
) -> None:
    if change.type in (ChangeType.rewrite, ChangeType.pos_tag, ChangeType.prune):
        after = [Sense.model_validate(s) for s in change.data["after"]]
        sense_store.update(
            change.form,
            lambda existing: existing.model_copy(  # type: ignore[union-attr]
                update={"senses": after}
            ),
        )
    elif change.type == ChangeType.morph_redirect:
        idx = change.data["derived_sense_idx"]
        after_sense = Sense.model_validate(change.data["after"])

        def apply_fn(existing: Alf | None) -> Alf:
            assert existing is not None
            senses = list(existing.senses)
            senses[idx] = after_sense
            return existing.model_copy(update={"senses": senses})

        sense_store.update(change.form, apply_fn)
    elif change.type == ChangeType.trim_sense:
        deleted_idx = change.data["deleted_idx"]
        after = [Sense.model_validate(s) for s in change.data["after"]]
        sense_store.update(
            change.form,
            lambda existing: existing.model_copy(  # type: ignore[union-attr]
                update={"senses": after}
            ),
        )
        if occ_store is not None:
            occ_store.delete_and_reindex_sense(change.form, deleted_idx)


def _examples_for_change(change: Change) -> list[list[str]]:
    """Return per-sense example snippets (rating>=3) for the change's form."""
    if _labeled is None or _docs is None:
        return []
    if change.type == ChangeType.morph_redirect:
        idx = change.data["derived_sense_idx"]
        return [
            fetch_instances(
                change.form,
                sense_key(idx),
                _labeled,
                _docs,
                min_rating=3,
                context_chars=60,
                max_instances=3,
                bold_form=True,
            )
        ]
    n_senses = len(change.data.get("before", []))
    return [
        fetch_instances(
            change.form,
            sense_key(i),
            _labeled,
            _docs,
            min_rating=3,
            context_chars=60,
            max_instances=3,
            bold_form=True,
        )
        for i in range(n_senses)
    ]


def _change_to_dict(change: Change) -> dict:  # type: ignore[type-arg]
    return {
        "id": change.id,
        "type": change.type.value,
        "form": change.form,
        "data": change.data,
        "examples": _examples_for_change(change),
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
    try:
        apply_change(change, _sense_store, _occ_store)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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


def main(
    senses_db: Path,
    changes_db: Path,
    labeled_db: Path | None = None,
    docs: Path | None = None,
    port: int = 5003,
) -> None:
    global _sense_store, _change_store, _occ_store, _labeled, _docs
    _sense_store = SenseStore(senses_db)
    _change_store = ChangeStore(changes_db)
    if labeled_db is not None and labeled_db.exists():
        _occ_store = OccurrenceStore(labeled_db)
        _labeled = _occ_store.to_polars()
    if docs is not None and docs.exists():
        _docs = pl.read_parquet(docs)
    app.run(host="127.0.0.1", port=port, debug=False)
