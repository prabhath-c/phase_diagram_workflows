"""
Unit tests for `refine_many_structures_with_discovered_tolerance` in
phase_diagram_workflows.free_energies.ts_convergence.many.

As in test_resubmission, calphy is stubbed (per structure and per `t_high` here) and the executor runs every
submitted function at once, in-process; the bracket logs, `tolerance.json` and the hand-overs are the real code.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

import phase_diagram_workflows.free_energies.ts_convergence.single as single_module
from phase_diagram_workflows.free_energies.ts_convergence.many import (
    load_discovered_tolerance,
    refine_many_structures_with_discovered_tolerance,
)
from phase_diagram_workflows.free_energies.ts_convergence.single import ExecutorSpec, load_bracket_history

from .test_resubmission import RecordingExecutor, SlurmClusterExecutor


class PerStructureCalphy:
    """calphy stand-in: the criterion is looked up by structure folder name and `t_high`."""

    def __init__(self, phase_directory, criteria):
        self.phase_directory = str(phase_directory)
        self.criteria = criteria    # {structure name: {t_high: criterion}}
        self.runs = []              # (structure name, t_high)
        self.finished = {}

    def name_of(self, working_directory):
        return os.path.relpath(working_directory, self.phase_directory).split(os.sep)[0]

    def calc(self, input_structure, potential_df, calphy_parameters, working_directory):
        name, t_high = self.name_of(working_directory), calphy_parameters["temperature"][1]
        self.runs.append((name, t_high))
        os.makedirs(working_directory, exist_ok=True)
        self.finished[working_directory] = self.criteria[name][t_high]

    def gather(self, working_directory):
        if working_directory not in self.finished:
            raise FileNotFoundError(working_directory)
        return pd.DataFrame([{
            "status": True,
            "forward_energy_diff": [np.array([0.0, 0.0])],
            "backward_energy_diff": [np.array([0.0, self.finished[working_directory]])],
        }])


@pytest.fixture
def phase(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # the executor cache directory defaults to a relative folder
    RecordingExecutor.instances.clear()
    RecordingExecutor.results.clear()

    def install(criteria):
        fake = PerStructureCalphy(tmp_path / "phase", criteria)
        monkeypatch.setattr(single_module, "calc_free_energy_with_calphy", fake.calc)
        monkeypatch.setattr(single_module, "gather_calphy_results_detailed", fake.gather)
        return fake

    return install


def arguments(tmp_path, names, executor_class=SlurmClusterExecutor, **overrides):
    arguments = dict(
        structures={name: bulk("Al", cubic=True) for name in names},
        calphy_parameters={"mode": "ts", "reference_phase": "solid", "temperature": [0, 0], "n_switching_steps": 80000},
        potential_df=pd.DataFrame({"Config": [["pair_style eam\n", "pair_coeff * *\n"]], "Species": [["Al"]]}),
        executor_spec=ExecutorSpec(executor_class),
        working_directory=str(tmp_path / "phase"),
        initial_bracket=(300.0, 1000.0),
        step_upper=100.0,
    )
    arguments.update(overrides)
    return arguments


# first brackets: a and b are noise, c and d are the spike; d needs one more step than c
CRITERIA = {
    "a": {1000.0: 0.0004},
    "b": {1000.0: 0.0007},
    "c": {1000.0: 0.040, 900.0: 0.0005},
    "d": {1000.0: 0.030, 900.0: 0.020, 800.0: 0.0008},
}


def test_tolerance_comes_from_the_first_brackets_and_only_the_failing_ones_are_narrowed(phase, tmp_path):
    fake = phase(CRITERIA)
    result = refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA))

    tolerance = load_discovered_tolerance(str(tmp_path / "phase"))
    assert 0.0007 < tolerance < 0.030
    assert tolerance == pytest.approx((0.0007 * 0.030) ** 0.5)
    assert result["first_brackets_submitted"] == ["a", "b", "c", "d"]
    assert sorted(fake.runs) == sorted([
        ("a", 1000.0), ("b", 1000.0), ("c", 1000.0), ("d", 1000.0), ("c", 900.0), ("d", 900.0), ("d", 800.0),
    ])
    for name, converged_at in [("a", 1000.0), ("b", 1000.0), ("c", 900.0), ("d", 800.0)]:
        _, criteria = load_bracket_history(str(tmp_path / "phase" / name))
        assert min(t_high for _, t_high in criteria) == converged_at


def test_every_structure_is_first_run_at_the_initial_bracket_with_no_tolerance_applied(phase, tmp_path):
    phase(CRITERIA)
    refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA))
    first_wave = RecordingExecutor.results[:4]   # a, b, c, d: accepted whatever they gave, nothing narrowed yet
    assert [outcome["status"] for outcome in first_wave] == ["converged"] * 4


def test_the_tolerance_is_stored_with_its_origin(phase, tmp_path):
    phase(CRITERIA)
    refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA))
    with open(tmp_path / "phase" / "tolerance.json") as handle:
        stored = json.load(handle)
    assert stored["method"] == "gap"
    assert stored["first_bracket_criteria"] == {"a": 0.0004, "b": 0.0007, "c": 0.040, "d": 0.030}


def test_the_tolerance_is_never_recomputed_from_later_brackets(phase, tmp_path):
    phase(CRITERIA)
    refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA))
    before = load_discovered_tolerance(str(tmp_path / "phase"))

    again = refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA))
    assert load_discovered_tolerance(str(tmp_path / "phase")) == before
    assert again["first_brackets_submitted"] == []
    assert again["narrowing_submitted"] == []    # everything converged already


def test_a_rerun_after_the_tolerance_is_known_resumes_only_the_narrowing(phase, tmp_path):
    fake = phase({**CRITERIA, "d": {1000.0: 0.030, 900.0: 0.020, 800.0: 0.0008}})
    refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA, max_iterations=0))
    # max_iterations=0: c and d stopped after their first bracket; nothing more was submitted
    assert sorted(fake.runs) == [("a", 1000.0), ("b", 1000.0), ("c", 1000.0), ("d", 1000.0)]
    runs_before = len(fake.runs)

    resumed = refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA))
    assert resumed["first_brackets_submitted"] == []
    assert resumed["narrowing_submitted"] == ["c", "d"]
    assert sorted(fake.runs[runs_before:]) == [("c", 900.0), ("d", 800.0), ("d", 900.0)]


def test_first_brackets_that_already_have_a_result_are_not_run_again(phase, tmp_path):
    fake = phase(CRITERIA)
    from phase_diagram_workflows.free_energies.ts_convergence.single import refine_temperature_bracket_with_resubmission

    settings = arguments(tmp_path, CRITERIA)
    refine_temperature_bracket_with_resubmission(     # "a" was run before, e.g. by an earlier attempt
        input_structure=settings["structures"]["a"], calphy_parameters=settings["calphy_parameters"],
        potential_df=settings["potential_df"], executor_spec=settings["executor_spec"],
        working_directory_root=str(tmp_path / "phase" / "a"), initial_bracket=(300.0, 1000.0),
        tolerance=np.inf, step_upper=100.0, max_iterations=0,
    )
    result = refine_many_structures_with_discovered_tolerance(**settings)
    assert result["first_brackets_submitted"] == ["b", "c", "d"]
    assert [run for run in fake.runs if run[1] == 1000.0].count(("a", 1000.0)) == 1


class TestNoGap:
    flat = {name: {1000.0: criterion} for name, criterion in zip("abcd", (0.0004, 0.0006, 0.0009, 0.0011))}

    def test_without_a_fallback_it_says_so_instead_of_guessing(self, phase, tmp_path):
        phase(self.flat)
        with pytest.raises(ValueError, match="fallback_tolerance"):
            refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, self.flat))
        assert load_discovered_tolerance(str(tmp_path / "phase")) is None

    def test_the_fallback_is_used_and_recorded(self, phase, tmp_path):
        fake = phase(self.flat)
        refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, self.flat, fallback_tolerance=0.002))
        assert load_discovered_tolerance(str(tmp_path / "phase")) == 0.002
        assert "fallback" in json.load(open(tmp_path / "phase" / "tolerance.json"))["method"]
        assert len(fake.runs) == 4     # all below 0.002: nothing narrowed


def test_a_first_bracket_without_a_result_stops_the_decision(phase, tmp_path, monkeypatch):
    phase(CRITERIA)
    monkeypatch.setattr(single_module, "_read_criterion", lambda working_directory: None)   # calphy "finished" without output
    with pytest.raises(RuntimeError):
        refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA))


def test_step_size_is_required(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA, step_upper=None))


def test_an_executor_that_does_not_detach_returns_when_everything_is_done(phase, tmp_path):
    phase(CRITERIA)
    result = refine_many_structures_with_discovered_tolerance(**arguments(tmp_path, CRITERIA, executor_class=RecordingExecutor))
    assert result["tolerance"] == pytest.approx((0.0007 * 0.030) ** 0.5)
    assert result["narrowing_submitted"] == ["c", "d"]
