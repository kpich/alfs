from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from alfs.actions import ACTIONS

from .queue_manager import QueueManager, Task

app = Flask(__name__)
_queue: QueueManager | None = None


def _task_to_dict(task: Task, est_duration: float | None = None) -> dict:
    return {
        "id": task.id,
        "type": task.type,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "ended_at": task.ended_at.isoformat() if task.ended_at else None,
        "returncode": task.returncode,
        "log_count": len(task.log_lines),
        "est_duration": est_duration,
    }


@app.get("/api/actions")
def list_actions():
    return jsonify(
        [
            {"name": a.name, "label": a.label, "human_review": a.human_review}
            for a in ACTIONS
        ]
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/tasks")
def list_tasks():
    assert _queue is not None
    return jsonify(
        [_task_to_dict(t, _queue.average_duration(t.type)) for t in _queue.all_tasks()]
    )


@app.post("/api/tasks")
def create_task():
    assert _queue is not None
    body = request.get_json(silent=True) or {}
    task_type = body.get("type", "")
    try:
        task = _queue.enqueue(task_type)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_task_to_dict(task, _queue.average_duration(task.type))), 201


@app.delete("/api/tasks/<task_id>")
def remove_task(task_id: str):
    assert _queue is not None
    if not _queue.remove_task(task_id):
        return jsonify({"error": "not found or not pending"}), 404
    return "", 204


@app.get("/api/tasks/<task_id>/logs")
def task_logs(task_id: str):
    assert _queue is not None
    task = _queue.get_task(task_id)
    if task is None:
        return jsonify({"error": "not found"}), 404
    from_idx = request.args.get("from", 0, type=int)
    lines = task.log_lines[from_idx:]
    return jsonify({"lines": lines, "total": len(task.log_lines)})


def main(project_root: Path, port: int = 5002) -> None:
    global _queue
    _queue = QueueManager(project_root)
    app.run(host="127.0.0.1", port=port, debug=False)
