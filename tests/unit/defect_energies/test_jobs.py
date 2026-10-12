"""Unit tests for defect_energies.jobs: the states of a job as files, and following them."""

import json
import os

import pytest

from phase_diagram_workflows.defect_energies import jobs
from phase_diagram_workflows.defect_energies.jobs import (
    ERROR_FILE,
    STARTED_FILE,
    follow_jobs,
    job_state,
    job_tracking,
    load_job_status,
    mark_submitted,
)


class TestStates:
    def test_a_job_goes_through_queued_running_done(self, tmp_path):
        folder = str(tmp_path / "a")
        assert job_state(folder)["state"] == "unknown"
        mark_submitted(folder)
        assert job_state(folder)["state"] == "queued"
        with job_tracking(folder):
            assert os.path.isfile(os.path.join(folder, STARTED_FILE))
            assert job_state(folder)["state"] == "running"
            open(os.path.join(folder, "result.pkl"), "w").close()
        assert job_state(folder)["state"] == "done"

    def test_an_exception_is_recorded_and_reraised(self, tmp_path):
        folder = str(tmp_path / "a")
        with pytest.raises(TypeError, match="atomid"):
            with job_tracking(folder):
                raise TypeError("create_atoms() got an unexpected keyword argument 'atomid'")
        state = job_state(folder)
        assert state["state"] == "failed" and "atomid" in state["error"] and "Traceback" in state["error"]

    def test_submitting_again_clears_the_old_failure(self, tmp_path):
        folder = str(tmp_path / "a")
        with pytest.raises(ValueError):
            with job_tracking(folder):
                raise ValueError("x")
        mark_submitted(folder)
        assert job_state(folder)["state"] == "queued"


class TestStatusTable:
    def test_lists_folders_with_status_files_and_accepts_both_result_names(self, tmp_path):
        mark_submitted(str(tmp_path / "queued"))
        os.makedirs(tmp_path / "relaxed_done")
        open(tmp_path / "relaxed_done" / "relaxed.pkl", "w").close()
        os.makedirs(tmp_path / "result_done")
        open(tmp_path / "result_done" / "result.pkl", "w").close()
        os.makedirs(tmp_path / "unrelated")
        table = load_job_status(str(tmp_path))
        assert dict(table["state"]) == {"queued": "queued", "relaxed_done": "done", "result_done": "done"}

    def test_names_selects_and_orders(self, tmp_path):
        mark_submitted(str(tmp_path / "b"))
        table = load_job_status(str(tmp_path), names=["a", "b"])
        assert list(table.index) == ["a", "b"] and list(table["state"]) == ["unknown", "queued"]


class TestFollowJobs:
    def test_prints_every_change_and_returns_when_all_finished(self, tmp_path, monkeypatch):
        folder = str(tmp_path / "a")
        mark_submitted(folder)
        steps = iter(["running", "done"])

        def finish(_):
            step = next(steps)
            if step == "running":
                with open(os.path.join(folder, STARTED_FILE), "w") as handle:
                    json.dump({}, handle)
            else:
                open(os.path.join(folder, "relaxed.pkl"), "w").close()

        monkeypatch.setattr(jobs.time, "sleep", finish)
        lines = []
        table = follow_jobs(str(tmp_path), names=["a"], printer=lines.append)
        assert table.loc["a", "state"] == "done"
        assert [line.split(": ")[1] for line in lines] == ["- -> queued", "queued -> running", "running -> done"]

    def test_failure_is_printed_and_raised_with_the_traceback(self, tmp_path, monkeypatch):
        folder = str(tmp_path / "a")
        mark_submitted(folder)
        with pytest.raises(KeyError):
            with job_tracking(folder):
                raise KeyError("boom")
        lines = []
        with pytest.raises(RuntimeError, match="boom"):
            follow_jobs(str(tmp_path), names=["a"], printer=lines.append)
        assert any("failed" in line for line in lines)
        assert follow_jobs(str(tmp_path), names=["a"], printer=lines.append, raise_on_failure=False).loc["a", "state"] == "failed"

    def test_timeout_returns_the_table_and_leaves_jobs_alone(self, tmp_path, monkeypatch):
        mark_submitted(str(tmp_path / "a"))
        monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
        table = follow_jobs(str(tmp_path), names=["a"], timeout=-1, printer=lambda line: None)
        assert table.loc["a", "state"] == "queued"
