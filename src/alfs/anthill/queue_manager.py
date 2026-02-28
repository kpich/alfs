from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import subprocess
import threading
import uuid


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


TASK_COMMANDS: dict[str, list[str]] = {
    "update": ["make", "update"],
}

MAX_LOG_LINES = 5000


@dataclass
class Task:
    id: str
    type: str
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    log_lines: list[str] = field(default_factory=list)
    returncode: int | None = None


class QueueManager:
    def __init__(self, project_root: Path, max_parallel: int = 4) -> None:
        self.project_root = project_root
        self.max_parallel = max_parallel
        self.tasks: list[Task] = []
        self._lock = threading.Lock()

        t = threading.Thread(target=self._dispatch_loop, daemon=True)
        t.start()

    def enqueue(self, task_type: str) -> Task:
        if task_type not in TASK_COMMANDS:
            raise ValueError(f"Unsupported task type: {task_type!r}")
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            status=TaskStatus.pending,
            created_at=datetime.utcnow(),
        )
        with self._lock:
            self.tasks.append(task)
        return task

    def get_task(self, task_id: str) -> Task | None:
        with self._lock:
            for t in self.tasks:
                if t.id == task_id:
                    return t
        return None

    def all_tasks(self) -> list[Task]:
        with self._lock:
            return list(self.tasks)

    def _dispatch_loop(self) -> None:
        import time

        while True:
            self._maybe_dispatch()
            time.sleep(0.5)

    def _maybe_dispatch(self) -> None:
        with self._lock:
            running = [t for t in self.tasks if t.status == TaskStatus.running]
            if len(running) >= self.max_parallel:
                return
            pending = [t for t in self.tasks if t.status == TaskStatus.pending]
            if not pending:
                return
            task = pending[0]
            task.status = TaskStatus.running
            task.started_at = datetime.utcnow()

        thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
        thread.start()

    def _run_task(self, task: Task) -> None:
        cmd = TASK_COMMANDS[task.type]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                with self._lock:
                    task.log_lines.append(line.rstrip("\n"))
                    if len(task.log_lines) > MAX_LOG_LINES:
                        task.log_lines = task.log_lines[-MAX_LOG_LINES:]
            proc.wait()
            with self._lock:
                task.returncode = proc.returncode
                task.status = (
                    TaskStatus.done if proc.returncode == 0 else TaskStatus.failed
                )
                task.ended_at = datetime.utcnow()
        except Exception as exc:
            with self._lock:
                task.log_lines.append(f"[conductor error] {exc}")
                task.status = TaskStatus.failed
                task.ended_at = datetime.utcnow()
