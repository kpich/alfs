from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import os
from pathlib import Path
import subprocess
import threading
import uuid

from alfs.actions import ACTIONS_BY_NAME

CC_TASKS_DIR_ENV = "CC_TASKS_DIR"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


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
    log_file: Path | None = None
    use_cc: bool = False


class QueueManager:
    def __init__(
        self, project_root: Path, max_parallel: int = 1, log_dir: Path | None = None
    ) -> None:
        # TODO: max_parallel > 1 will cause nextflow lock collisions — all pipelines
        # run from the same launch dir and share .nextflow.lock. Also currently
        # GPU-bound on MPS so parallelism doesn't help.
        self.project_root = project_root
        self.max_parallel = max_parallel
        self.log_dir = log_dir
        self.tasks: list[Task] = []
        self._lock = threading.Lock()
        self._durations: dict[str, list[float]] = {}

        t = threading.Thread(target=self._dispatch_loop, daemon=True)
        t.start()

    def enqueue(self, task_type: str, use_cc: bool = False) -> Task:
        if task_type not in ACTIONS_BY_NAME:
            raise ValueError(f"Unsupported task type: {task_type!r}")
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            status=TaskStatus.pending,
            created_at=datetime.now(UTC),
            use_cc=use_cc,
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

    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            for i, t in enumerate(self.tasks):
                if t.id == task_id and t.status == TaskStatus.pending:
                    del self.tasks[i]
                    return True
        return False

    def average_duration(self, task_type: str) -> float | None:
        vals = self._durations.get(task_type)
        return sum(vals) / len(vals) if vals else None

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
            task.started_at = datetime.now(UTC)

        thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
        thread.start()

    def _run_task(self, task: Task) -> None:
        cmd = ACTIONS_BY_NAME[task.type].cmd
        try:
            log_path: Path | None = None
            if self.log_dir:
                date_str = datetime.now(UTC).strftime("%Y-%m-%d")
                ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
                fname = f"{task.type}_{ts}_{task.id[:8]}.log"
                log_path = self.log_dir / date_str / fname
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._lock:
                    task.log_file = log_path

            env = None
            if task.use_cc:
                cc_dir = os.environ.get(CC_TASKS_DIR_ENV, "../cc_tasks")
                env = {
                    **os.environ,
                    CC_TASKS_DIR_ENV: cc_dir,
                }
            proc = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            assert proc.stdout is not None
            with contextlib.ExitStack() as stack:
                log_fh = stack.enter_context(open(log_path, "w")) if log_path else None
                for line in proc.stdout:
                    with self._lock:
                        task.log_lines.append(line.rstrip("\n"))
                        if len(task.log_lines) > MAX_LOG_LINES:
                            task.log_lines = task.log_lines[-MAX_LOG_LINES:]
                    if log_fh is not None:
                        log_fh.write(line)
                        log_fh.flush()
                proc.wait()
                if log_fh is not None:
                    log_fh.write(f"\n[exit {proc.returncode}]\n")
            with self._lock:
                task.returncode = proc.returncode
                task.status = (
                    TaskStatus.done if proc.returncode == 0 else TaskStatus.failed
                )
                task.ended_at = datetime.now(UTC)
                elapsed = (
                    (task.ended_at - task.started_at).total_seconds()
                    if task.started_at
                    else 0.0
                )
                self._durations.setdefault(task.type, []).append(elapsed)
        except Exception as exc:
            with self._lock:
                task.log_lines.append(f"[conductor error] {exc}")
                task.status = TaskStatus.failed
                task.ended_at = datetime.now(UTC)
