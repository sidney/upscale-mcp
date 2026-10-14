"""JobManager tests with plain functions standing in for a model run.

The fake work functions block on threading.Events the test controls, so the
tests decide exactly when a job is "in the middle of a tile" instead of
relying on sleeps and timing.
"""

import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from upscale_mcp import engine
from upscale_mcp.jobs import JobManager


def _work(total: int, gate: threading.Event | None = None, started: threading.Event | None = None, fail: bool = False):
    """A fake job: reports `total` tiles, optionally pausing before the first one."""

    def work(progress):
        if started is not None:
            started.set()
        if gate is not None:
            assert gate.wait(5), "test gate never opened"
        for i in range(1, total + 1):
            progress(i, total)
        if fail:
            raise ValueError("model exploded")
        return {"tiles": total}

    return work


def test_job_runs_to_done(tmp_path: Path) -> None:
    jobs = JobManager()
    job = jobs.submit(input="a.png", model="m", output=tmp_path / "a.out.png", work=_work(3))
    assert jobs.wait(job, 5)
    snap = job.snapshot()
    assert snap["status"] == "done"
    assert snap["result"] == {"tiles": 3}
    assert snap["progress"] == {"tiles_done": 3, "tiles_total": 3}
    assert snap["running_seconds"] is not None


def test_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    jobs = JobManager()
    job = jobs.submit(input="a.png", model="m", output=tmp_path / "a.png", work=_work(2, fail=True))
    assert jobs.wait(job, 5)
    assert job.status == "failed"
    assert job.error == "ValueError: model exploded"


def test_jobs_run_one_at_a_time_and_queued_job_can_be_cancelled(tmp_path: Path) -> None:
    jobs = JobManager()
    gate, started = threading.Event(), threading.Event()
    first = jobs.submit(input="a", model="m", output=tmp_path / "1.png", work=_work(2, gate, started))
    assert started.wait(5)  # first job holds the GPU lock, waiting on the gate
    second = jobs.submit(input="b", model="m", output=tmp_path / "2.png", work=_work(2))
    assert not jobs.wait(second, 0.2)  # blocked behind the first
    assert second.status == "queued"

    jobs.cancel(second.id)
    assert second.status == "cancelled" and second.finished.is_set()

    gate.set()
    assert jobs.wait(first, 5) and first.status == "done"
    assert second.result is None  # never ran


def test_running_job_stops_at_next_tile(tmp_path: Path) -> None:
    jobs = JobManager()
    gate, started = threading.Event(), threading.Event()
    job = jobs.submit(input="a", model="m", output=tmp_path / "1.png", work=_work(10, gate, started))
    assert started.wait(5)
    assert job.status == "running"
    jobs.cancel(job.id)
    gate.set()
    assert jobs.wait(job, 5)
    assert job.status == "cancelled"
    assert job.done == 1  # the first progress call raised


def test_same_output_refused_while_active(tmp_path: Path) -> None:
    jobs = JobManager()
    gate, started = threading.Event(), threading.Event()
    out = tmp_path / "same.png"
    first = jobs.submit(input="a", model="m", output=out, work=_work(1, gate, started))
    assert started.wait(5)
    with pytest.raises(FileExistsError):
        jobs.submit(input="b", model="m", output=out, work=_work(1))
    gate.set()
    assert jobs.wait(first, 5)
    # Once the first job has finished, the same path is free again (the
    # overwrite check against the file itself happens in Roots, not here).
    second = jobs.submit(input="b", model="m", output=out, work=_work(1))
    assert jobs.wait(second, 5) and second.status == "done"


def test_list_and_unknown_job(tmp_path: Path) -> None:
    jobs = JobManager(keep_finished=2)
    for i in range(4):
        j = jobs.submit(input=str(i), model="m", output=tmp_path / f"{i}.png", work=_work(1))
        assert jobs.wait(j, 5)
    listed = jobs.list()
    assert [s["input"] for s in listed] == ["3", "2"]  # newest first, oldest pruned
    with pytest.raises(ValueError):
        jobs.get("j1")


def test_save_image_is_atomic_and_leaves_no_temp(tmp_path: Path) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    dst = tmp_path / "out.png"
    engine.save_image(dst, rgb, None, None)
    assert Image.open(dst).size == (8, 8)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.png"]
    with pytest.raises(ValueError):
        engine.save_image(tmp_path / "out.bmp", rgb, None, None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.png"]
