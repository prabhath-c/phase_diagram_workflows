"""
Status of the jobs of a run, kept as files next to the results, so failures are never silent.

Every job of ``relax_structures`` and ``converge_formation_energy`` has a folder. The notebook writes
``submitted.json`` into it when it submits the job; the job itself writes ``started.json`` when it begins
(with its SLURM job id) and, if it raises, ``error.txt`` with the traceback; a finished job has its result file.
That gives every job one of these states:

``queued`` (submitted, not started), ``running``, ``done`` and ``failed`` (traceback in the table). A job that
stays ``queued`` or ``running`` for much longer than it should is the sign that something is wrong with it.

``load_job_status`` reads the states at any moment, from any process. ``follow_jobs`` waits for the jobs and
prints every change as it happens, so the notebook cell that submitted them shows what is going on.
The files do not depend on executorlib's cache.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
import traceback
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import pandas as pd

SUBMITTED_FILE = "submitted.json"
STARTED_FILE = "started.json"
ERROR_FILE = "error.txt"
TERMINAL_STATES = ("done", "failed")
RESULT_FILES = ("relaxed.pkl", "result.pkl")   # what means "done": relax_structures, converge_formation_energy


def _as_tuple(result_file: Union[str, Sequence[str]]) -> tuple:
    return (result_file,) if isinstance(result_file, str) else tuple(result_file)


def _write_json(path: str, content: Dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(content, handle)


def mark_submitted(folder: str) -> None:
    """Called by the notebook when it submits the job of `folder`; clears what an earlier attempt left."""
    os.makedirs(folder, exist_ok=True)
    for stale in (STARTED_FILE, ERROR_FILE):
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(folder, stale))
    _write_json(os.path.join(folder, SUBMITTED_FILE), {"time": time.time()})


@contextlib.contextmanager
def job_tracking(folder: str) -> Iterator[None]:
    """Wrap the body of a job: writes ``started.json`` first and ``error.txt`` (then re-raises) if it fails."""
    os.makedirs(folder, exist_ok=True)
    _write_json(
        os.path.join(folder, STARTED_FILE),
        {"time": time.time(), "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
    )
    try:
        yield
    except BaseException:
        with open(os.path.join(folder, ERROR_FILE), "w") as handle:
            handle.write(traceback.format_exc())
        raise


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def job_state(folder: str, result_file: Union[str, Sequence[str]] = RESULT_FILES) -> Dict[str, Any]:
    """State of the job of `folder`: ``{"state", "error", "slurm_job_id"}`` (see the module docstring)."""
    started_path = os.path.join(folder, STARTED_FILE)
    started = os.path.isfile(started_path)
    job_id = _read_json(started_path).get("slurm_job_id")
    if any(os.path.isfile(os.path.join(folder, f)) for f in _as_tuple(result_file)):
        return {"state": "done", "error": None, "slurm_job_id": job_id}
    error_path = os.path.join(folder, ERROR_FILE)
    if os.path.isfile(error_path):
        with open(error_path) as handle:
            return {"state": "failed", "error": handle.read(), "slurm_job_id": job_id}
    if started:
        return {"state": "running", "error": None, "slurm_job_id": job_id}
    if os.path.isfile(os.path.join(folder, SUBMITTED_FILE)):
        return {"state": "queued", "error": None, "slurm_job_id": None}
    return {"state": "unknown", "error": None, "slurm_job_id": None}


def load_job_status(
    working_directory: str, result_file: Union[str, Sequence[str]] = RESULT_FILES, names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Table of the jobs in the sub-folders of `working_directory`: ``state``, ``error`` (traceback) and ``slurm_job_id``.

    `result_file` is the file that means "done"; the default accepts the result of either module (``relaxed.pkl`` or
    ``result.pkl``). Without `names`, every
    sub-folder that has a status file or a result is listed; with `names`, exactly those, in that order.
    """
    if names is None:
        names = []
        for name in sorted(os.listdir(working_directory)) if os.path.isdir(working_directory) else []:
            folder = os.path.join(working_directory, name)
            if os.path.isdir(folder) and any(
                os.path.isfile(os.path.join(folder, f)) for f in (SUBMITTED_FILE, STARTED_FILE, ERROR_FILE, *_as_tuple(result_file))
            ):
                names.append(name)
    rows = {name: job_state(os.path.join(working_directory, name), result_file) for name in names}
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["state", "error", "slurm_job_id"])
    table.index.name = "name"
    return table


def follow_jobs(
    working_directory: str,
    names: Sequence[str],
    result_file: Union[str, Sequence[str]] = RESULT_FILES,
    interval: float = 15.0,
    raise_on_failure: bool = True,
    timeout: Optional[float] = None,
    printer=print,
) -> pd.DataFrame:
    """Wait for the jobs `names` and print every state change, so the submitting cell shows the progress.

    Returns the status table once every job is ``done`` or ``failed``. Interrupting the cell stops
    the waiting only; the SLURM jobs go on. With `raise_on_failure`, a ``RuntimeError`` with the traceback of the
    first failure is raised at the end, after the table was printed.
    """
    start = time.time()
    last: Dict[str, str] = {}
    table = load_job_status(working_directory, result_file, names)
    try:
        while True:
            table = load_job_status(working_directory, result_file, names)
            for name, row in table.iterrows():
                if last.get(name) != row["state"]:
                    note = ""
                    if row["state"] == "failed":
                        note = ": " + (row["error"].strip().splitlines() or [""])[-1]
                    printer(f"[{time.strftime('%H:%M:%S')}] {name}: {last.get(name, '-')} -> {row['state']}{note}")
                    last[name] = row["state"]
            if table["state"].isin(TERMINAL_STATES).all():
                break
            if timeout is not None and time.time() - start > timeout:
                printer(f"stopped waiting after {timeout:.0f} s; the jobs keep running")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        printer("stopped waiting; the jobs keep running")
        return table
    failed = table[table["state"] == "failed"]
    if raise_on_failure and len(failed):
        first = failed.iloc[0]
        raise RuntimeError(f"{len(failed)} of {len(table)} jobs failed; first, {failed.index[0]}:\n{first['error']}")
    return table
