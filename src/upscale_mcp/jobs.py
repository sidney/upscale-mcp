"""Background jobs for upscale-mcp.

Why this exists: some MCP clients give up on a tool call long before a slow
model finishes. A cloud session reaching the Mac through the Claude desktop
app waits only 60 seconds, while a DAT-class model can take minutes on a full
photo. The job itself carried on and wrote its output, but the caller saw a
timeout and had no way to follow it.

So every upscale now runs as a job on its own thread:

    submit()  -> Job (status "queued")
    thread waits for the GPU lock -> "running", progress updated per tile
              -> "done" (result) | "failed" (error) | "cancelled"

The MCP layer can wait for the job (fully, or up to a deadline) or return the
job id immediately and let the caller poll with job_status. Nothing here
knows about MCP or torch, so the tests can drive it with plain functions.

Jobs run one at a time (one GPU lock); queued jobs wait on the lock. Python's
Lock does not promise first-come-first-served, so with several jobs queued
the order they start in is not guaranteed. Threads are daemons, so a
running job dies with the server process (Claude Desktop quitting); outputs
are written atomically by engine.save_image, so a killed job leaves no
half-written file under the final name.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("upscale_mcp.jobs")

ProgressFn = Callable[[int, int], None]
WorkFn = Callable[[ProgressFn], dict]

ACTIVE = ("queued", "running")
FINISHED = ("done", "failed", "cancelled")


class JobCancelled(Exception):
    """Raised from the progress callback to stop a running job between tiles."""


@dataclass
class Job:
    id: str
    input: str
    model: str
    output: str
    status: str = "queued"
    done: int = 0  # tiles finished
    total: int = 0  # tiles in the job (0 until the first tile reports)
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    finished: threading.Event = field(default_factory=threading.Event, repr=False)

    def snapshot(self) -> dict:
        """JSON-friendly view of the job, safe to return from a tool."""
        now = time.time()
        end = self.finished_at or now
        return {
            "job": self.id,
            "status": self.status,
            "input": self.input,
            "model": self.model,
            "output": self.output,
            "progress": {"tiles_done": self.done, "tiles_total": self.total},
            "queued_seconds": round((self.started_at or end) - self.queued_at, 1),
            "running_seconds": round(end - self.started_at, 1) if self.started_at else None,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self, keep_finished: int = 50) -> None:
        self._lock = threading.Lock()  # guards _jobs and every status change
        self._gpu = threading.Lock()  # one job on the device at a time
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._ids = itertools.count(1)
        self._keep_finished = keep_finished

    # -- submitting -------------------------------------------------------

    def submit(self, *, input: str, model: str, output: Path, work: WorkFn) -> Job:
        """Queue `work` and start its thread. `work(progress)` returns the result dict.

        Refuses a second active job writing the same output path: both would
        pass the "file already exists" check at submit time and the second
        would silently replace the first's result.
        """
        out = str(output)
        with self._lock:
            for other in self._jobs.values():
                if other.status in ACTIVE and other.output == out:
                    raise FileExistsError(f"job {other.id} is already writing {out}")
            job = Job(id=f"j{next(self._ids)}", input=input, model=model, output=out)
            self._jobs[job.id] = job
            self._prune()
        threading.Thread(target=self._run, args=(job, work), name=f"upscale-{job.id}", daemon=True).start()
        return job

    def _run(self, job: Job, work: WorkFn) -> None:
        with self._gpu:
            with self._lock:
                if job.status == "cancelled":  # cancelled while queued
                    return
                job.status = "running"
                job.started_at = time.time()

            def progress(done: int, total: int) -> None:
                job.done, job.total = done, total
                if job.cancel_requested.is_set():
                    raise JobCancelled()

            try:
                result = work(progress)
            except JobCancelled:
                self._finish(job, "cancelled")
            except Exception as exc:
                log.exception("job %s failed", job.id)
                self._finish(job, "failed", error=f"{type(exc).__name__}: {exc}")
            else:
                self._finish(job, "done", result=result)

    def _finish(self, job: Job, status: str, *, result: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            job.status = status
            job.result = result
            job.error = error
            job.finished_at = time.time()
            # Prune here as well as on submit, so the retained history never
            # exceeds keep_finished once things go quiet.
            self._prune()
        job.finished.set()
        log.info("job %s %s", job.id, status)

    def _prune(self) -> None:
        """Forget the oldest finished jobs beyond keep_finished (caller holds _lock)."""
        finished = [j for j in self._jobs.values() if j.status in FINISHED]
        for j in finished[: max(0, len(finished) - self._keep_finished)]:
            del self._jobs[j.id]

    # -- querying and control ---------------------------------------------

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"no job {job_id!r} (finished jobs are kept only for the {self._keep_finished} most recent)")
        return job

    def list(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.snapshot() for j in reversed(jobs)]  # newest first

    def cancel(self, job_id: str) -> Job:
        """Cancel a queued job at once, or ask a running one to stop after its current tile."""
        job = self.get(job_id)
        with self._lock:
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                job.finished.set()
                return job
        if job.status == "running":
            job.cancel_requested.set()
        return job

    def wait(self, job: Job, timeout: float | None) -> bool:
        """Block until the job finishes or `timeout` seconds pass; True if finished."""
        return job.finished.wait(timeout)
