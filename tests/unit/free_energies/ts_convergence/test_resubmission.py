"""
Unit tests for `refine_temperature_bracket_with_resubmission` (and ExecutorSpec) in
phase_diagram_workflows.free_energies.ts_convergence.single.

calphy itself is stubbed at the two calls `single` makes to it (`calc_free_energy_with_calphy` and
`gather_calphy_results_detailed`); everything else -- the bracket log, start/submit markers,
configuration pinning, resubmission through an ExecutorSpec -- is the real code acting on a real
temporary directory. The fake calphy returns a criterion looked up from a per-`t_high` schedule,
so each test states exactly which brackets converge.
"""

import collections
import inspect
import os
import pickle
import time

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

import phase_diagram_workflows.free_energies.ts_convergence.single as single_module
from phase_diagram_workflows.free_energies.ts_convergence.single import (
    ExecutorSpec,
    current_bracket_resource_dict,
    load_bracket_history,
    refine_temperature_bracket_with_resubmission,
)


class RecordingExecutor:
    """Runs each submitted function immediately, in-process, and records how it was used."""

    instances = []
    results = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.submitted = 0
        self.resource_dicts = []
        self.shutdown_wait = None
        RecordingExecutor.instances.append(self)

    def submit(self, fn, resource_dict=None, **kwargs):
        from concurrent.futures import Future

        self.submitted += 1
        self.resource_dicts.append(resource_dict)
        future = Future()
        result = fn(**kwargs)  # exceptions propagate to the test, not into a Future
        RecordingExecutor.results.append(result)
        future.set_result(result)
        return future

    def shutdown(self, wait=True, **kwargs):
        self.shutdown_wait = wait


class SlurmClusterExecutor(RecordingExecutor):
    """Same behaviour; the class *name* is what makes ExecutorSpec treat it as detachable."""


class _Refusing(RecordingExecutor):
    def submit(self, fn, **kwargs):
        raise RuntimeError("queue unreachable")


class Calphy:
    """Stand-in for calphy: which brackets converge is decided by `criteria[t_high]`."""

    def __init__(self, criteria):
        self.criteria = criteria
        self.runs = []  # (t_low, t_high, n_switching_steps)
        self.finished = {}  # working_directory -> criterion

    def calc(self, input_structure, potential_df, calphy_parameters, working_directory):
        t_low, t_high = calphy_parameters["temperature"]
        self.runs.append((t_low, t_high, calphy_parameters.get("n_switching_steps")))
        os.makedirs(working_directory, exist_ok=True)
        open(os.path.join(working_directory, "log.lammps"), "w").close()
        self.finished[working_directory] = self.criteria[t_high]

    def gather(self, working_directory):
        if working_directory not in self.finished:
            raise FileNotFoundError(working_directory)
        criterion = self.finished[working_directory]
        return pd.DataFrame(
            [{
                "status": True,
                "forward_energy_diff": [np.array([0.0, 0.0])],
                "backward_energy_diff": [np.array([0.0, criterion])],
            }]
        )


@pytest.fixture
def calphy(monkeypatch):
    def install(criteria):
        fake = Calphy(criteria)
        monkeypatch.setattr(single_module, "calc_free_energy_with_calphy", fake.calc)
        monkeypatch.setattr(single_module, "gather_calphy_results_detailed", fake.gather)
        return fake

    RecordingExecutor.instances.clear()
    RecordingExecutor.results.clear()
    return install


@pytest.fixture
def setup(tmp_path):
    return dict(
        input_structure=bulk("Al", cubic=True),
        calphy_parameters={"mode": "ts", "reference_phase": "solid", "temperature": [0, 0], "n_switching_steps": 80000},
        potential_df=pd.DataFrame({"Config": [["pair_style eam\n", "pair_coeff * *\n"]], "Species": [["Al"]]}),
        executor_spec=ExecutorSpec(RecordingExecutor),
        working_directory_root=str(tmp_path / "structure"),
        initial_bracket=(300.0, 1000.0),
        tolerance=0.0075,
        step_upper=100.0,
    )


class TestExecutorSpec:
    def test_create_builds_a_fresh_executor_from_the_recipe(self):
        spec = ExecutorSpec(RecordingExecutor, {"queue": "cmmg"})
        first, second = spec.create(), spec.create()
        assert first is not second
        assert first.kwargs == {"queue": "cmmg"}

    def test_spec_itself_can_be_pickled(self):
        spec = ExecutorSpec(RecordingExecutor, {"resource_dict": {"cores": 1}})
        assert pickle.loads(pickle.dumps(spec)) == spec

    def test_unpicklable_setting_fails_at_construction(self):
        with pytest.raises(ValueError, match="cannot be pickled"):
            ExecutorSpec(RecordingExecutor, {"callback": lambda: None})

    def test_detaches_only_for_executors_that_survive_their_submitter(self):
        assert ExecutorSpec(SlurmClusterExecutor).detaches is True
        assert ExecutorSpec(RecordingExecutor).detaches is False
        assert ExecutorSpec(RecordingExecutor, detach=True).detaches is True
        assert ExecutorSpec(SlurmClusterExecutor, detach=False).detaches is False


class TestNarrowing:
    def test_converges_on_first_bracket_without_submitting_anything(self, calphy, setup):
        fake = calphy({1000.0: 0.001})
        result = refine_temperature_bracket_with_resubmission(**setup)

        assert result["status"] == "converged"
        assert result["bracket"] == (300.0, 1000.0)
        assert [run[:2] for run in fake.runs] == [(300.0, 1000.0)]
        assert RecordingExecutor.instances == []

    def test_narrows_until_converged_each_step_submitted_through_the_spec(self, calphy, setup):
        fake = calphy({1000.0: 0.03, 900.0: 0.02, 800.0: 0.001})
        result = refine_temperature_bracket_with_resubmission(**setup)

        assert result["status"] == "resubmitted"  # the outermost call only handed on
        assert [run[:2] for run in fake.runs] == [(300.0, 1000.0), (300.0, 900.0), (300.0, 800.0)]
        tried, criteria = load_bracket_history(setup["working_directory_root"])
        assert sorted(tried) == [(300.0, 800.0), (300.0, 900.0), (300.0, 1000.0)]
        assert criteria == {(300.0, 1000.0): 0.03, (300.0, 900.0): 0.02, (300.0, 800.0): 0.001}
        assert len(RecordingExecutor.instances) == 2  # one fresh executor per hand-over

    def test_collapsing_bracket_ends_as_exhausted_instead_of_raising(self, calphy, setup):
        fake = calphy({720.0: 1.0, 710.0: 1.0})
        setup.update(initial_bracket=(700.0, 720.0), step_upper=10.0)
        refine_temperature_bracket_with_resubmission(**setup)

        assert [run[:2] for run in fake.runs] == [(700.0, 720.0), (700.0, 710.0)]
        last = refine_temperature_bracket_with_resubmission(**setup)  # a re-run just reports where it ended
        assert last["status"] == "exhausted"
        assert last["bracket"] == (700.0, 710.0)

    def test_step_size_is_required(self, setup):
        setup.pop("step_upper")
        with pytest.raises(ValueError, match="At least one"):
            refine_temperature_bracket_with_resubmission(**setup)

    def test_switching_steps_scale_with_bracket_width_when_asked(self, calphy, setup):
        fake = calphy({1000.0: 1.0, 500.0: 0.001})
        setup.update(step_upper=500.0, scale_switching_steps_with_range=True)
        refine_temperature_bracket_with_resubmission(**setup)
        assert [run[2] for run in fake.runs] == [80000, 22857]  # 80000 * 200/700

    def test_switching_steps_stay_constant_when_scaling_is_off(self, calphy, setup):
        fake = calphy({1000.0: 1.0, 500.0: 0.001})
        setup.update(step_upper=500.0, scale_switching_steps_with_range=False)
        refine_temperature_bracket_with_resubmission(**setup)
        assert [run[2] for run in fake.runs] == [80000, 80000]


class TestTaskName:
    """The executorlib task name is ours: structure folder + bracket, never a hash of the arguments."""

    @pytest.fixture
    def cached(self, tmp_path, setup):
        cache = tmp_path / "cache"
        cache.mkdir()
        setup["executor_spec"] = ExecutorSpec(RecordingExecutor, {"cache_directory": str(cache)})
        return cache

    def stale_result(self, cache, setup, bracket):
        path = cache / (single_module._bracket_cache_key(setup["working_directory_root"], bracket) + "_o.h5")
        path.write_text("result of an earlier attempt")
        return path

    def test_new_folder_is_named_after_the_initial_bracket(self, setup):
        name = current_bracket_resource_dict(setup["working_directory_root"], setup["initial_bracket"], setup["executor_spec"])
        assert name == {"cache_key": single_module._bracket_cache_key(setup["working_directory_root"], (300.0, 1000.0))}
        assert "T_300.00_1000.00" in name["cache_key"]

    def test_same_bracket_of_different_structures_has_different_names(self):
        bracket = (300.0, 900.0)
        assert single_module._bracket_cache_key("/a/c_0d2", bracket) != single_module._bracket_cache_key("/a/c_0d4", bracket)

    def test_name_does_not_depend_on_the_arguments_of_the_job(self, setup):
        # the old hash-of-arguments name changed with any of these; ours must not
        first = current_bracket_resource_dict(setup["working_directory_root"], setup["initial_bracket"], setup["executor_spec"])
        again = current_bracket_resource_dict(setup["working_directory_root"], setup["initial_bracket"], ExecutorSpec(RecordingExecutor, {"a": 1}))
        assert first == again

    def test_a_leftover_result_is_removed_and_other_files_are_kept(self, cached, setup):
        stale = self.stale_result(cached, setup, (300.0, 1000.0))
        other = cached / "unrelated_o.h5"
        other.write_text("x")
        current_bracket_resource_dict(setup["working_directory_root"], setup["initial_bracket"], setup["executor_spec"])
        assert not stale.exists()
        assert other.exists()

    def test_missing_cache_directory_is_fine(self, setup):
        setup["executor_spec"] = ExecutorSpec(RecordingExecutor, {"cache_directory": "/nonexistent/place"})
        current_bracket_resource_dict(setup["working_directory_root"], setup["initial_bracket"], setup["executor_spec"])

    def test_name_follows_the_bracket_log(self, calphy, cached, setup):
        calphy({1000.0: 0.03, 900.0: 0.02})
        setup["max_iterations"] = 1
        refine_temperature_bracket_with_resubmission(**setup)  # brackets 1000 and 900 ran, then the cap
        name = current_bracket_resource_dict(setup["working_directory_root"], setup["initial_bracket"], setup["executor_spec"])
        assert "T_300.00_900.00" in name["cache_key"]

    def test_continuing_after_the_cap_is_not_blocked_by_the_cached_result_of_that_bracket(self, calphy, cached, setup):
        fake = calphy({1000.0: 0.03, 900.0: 0.02, 800.0: 0.001})
        setup["max_iterations"] = 1
        refine_temperature_bracket_with_resubmission(**setup)
        stale = self.stale_result(cached, setup, (300.0, 900.0))  # its job ended with 'limit_reached'

        resource_dict = current_bracket_resource_dict(setup["working_directory_root"], setup["initial_bracket"], setup["executor_spec"])
        assert not stale.exists()  # so executorlib will submit rather than answer from its cache
        assert resource_dict["cache_key"] == single_module._bracket_cache_key(setup["working_directory_root"], (300.0, 900.0))

        setup["max_iterations"] = 3
        result = refine_temperature_bracket_with_resubmission(**setup)
        assert [run[:2] for run in fake.runs][-1] == (300.0, 800.0)
        assert result["status"] == "resubmitted"

    def test_retry_after_a_failed_job_is_not_blocked_by_its_error_result(self, calphy, cached, setup):
        calphy({1000.0: 0.03, 900.0: 0.02, 800.0: 0.001})
        setup["max_iterations"] = 3
        stale = self.stale_result(cached, setup, (300.0, 900.0))  # a failed job's result for the next bracket
        refine_temperature_bracket_with_resubmission(**setup)
        assert not stale.exists()  # cleared by the hand-over before it submitted 900 K


class TestHandOver:
    def test_every_bracket_gets_its_own_task_name(self, calphy, setup):
        # regression: without it, job 3 had the same executorlib task name as the still-running
        # job 2 and the SLURM executor silently declined to submit it
        calphy({1000.0: 0.03, 900.0: 0.02, 800.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        keys = [executor.resource_dicts[0]["cache_key"] for executor in RecordingExecutor.instances]
        assert len(keys) == 2
        assert len(set(keys)) == 2
        assert "T_300.00_900.00" in keys[0] and "T_300.00_800.00" in keys[1]

    def test_detachable_executor_is_shut_down_without_waiting(self, calphy, setup):
        calphy({1000.0: 1.0, 900.0: 0.001})
        setup["executor_spec"] = ExecutorSpec(SlurmClusterExecutor)
        refine_temperature_bracket_with_resubmission(**setup)
        assert [executor.shutdown_wait for executor in RecordingExecutor.instances] == [False]

    def test_non_detachable_executor_is_waited_for(self, calphy, setup):
        calphy({1000.0: 1.0, 900.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        assert [executor.shutdown_wait for executor in RecordingExecutor.instances] == [True]

    def test_bracket_already_handed_on_is_not_submitted_a_second_time(self, calphy, setup):
        calphy({1000.0: 1.0, 900.0: 0.001})
        os.makedirs(setup["working_directory_root"])
        open(single_module._marker_path(setup["working_directory_root"], "submitted", (300.0, 900.0)), "w").close()

        refine_temperature_bracket_with_resubmission(**setup)
        assert RecordingExecutor.instances == []

    def test_failed_submission_releases_the_claim_so_it_can_be_retried(self, calphy, setup):
        calphy({1000.0: 1.0, 900.0: 0.001})

        setup["executor_spec"] = ExecutorSpec(_Refusing)
        with pytest.raises(RuntimeError, match="queue unreachable"):
            refine_temperature_bracket_with_resubmission(**setup)
        assert not os.path.exists(single_module._marker_path(setup["working_directory_root"], "submitted", (300.0, 900.0)))


class TestResume:
    def test_rerun_skips_finished_brackets_and_continues_from_the_first_without_result(self, calphy, setup):
        fake = calphy({1000.0: 0.03, 900.0: 0.001})
        # a previous run: 1000 K finished; its successor (900 K) was submitted but its job died
        refine_temperature_bracket_with_resubmission(**{**setup, "executor_spec": ExecutorSpec(_Dying, detach=True)})
        assert [run[:2] for run in fake.runs] == [(300.0, 1000.0)]
        tried, criteria = load_bracket_history(setup["working_directory_root"])
        assert (300.0, 900.0) in tried and (300.0, 900.0) not in criteria

        fake.runs.clear()
        result = refine_temperature_bracket_with_resubmission(**{**setup, "stale_after_seconds": 0.0})
        assert [run[:2] for run in fake.runs] == [(300.0, 900.0)]  # 1000 K not run again
        assert result["status"] == "converged"

    def test_already_converged_folder_runs_nothing(self, calphy, setup):
        fake = calphy({1000.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        fake.runs.clear()

        result = refine_temperature_bracket_with_resubmission(**setup)
        assert result["status"] == "converged"
        assert fake.runs == []

    def test_looser_tolerance_reuses_a_wider_bracket_that_already_qualifies(self, calphy, setup):
        fake = calphy({1000.0: 0.03, 900.0: 0.02, 800.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        fake.runs.clear()

        result = refine_temperature_bracket_with_resubmission(**{**setup, "tolerance": 0.05})
        assert (result["status"], result["bracket"]) == ("converged", (300.0, 1000.0))
        assert fake.runs == []

    def test_result_that_finished_on_disk_but_was_never_logged_is_picked_up(self, calphy, setup):
        fake = calphy({1000.0: 0.001})
        working_directory = os.path.join(setup["working_directory_root"], "T_300.00_1000.00")
        fake.finished[working_directory] = 0.001  # calphy output exists, log has no row

        result = refine_temperature_bracket_with_resubmission(**setup)
        assert result["status"] == "converged"
        assert fake.runs == []
        assert load_bracket_history(setup["working_directory_root"])[1] == {(300.0, 1000.0): 0.001}


class _Dying(RecordingExecutor):
    """Accepts the successor submission but never runs it (a job that died in the queue)."""

    def submit(self, fn, **kwargs):
        from concurrent.futures import Future

        return Future()

    def shutdown(self, wait=True, **kwargs):
        pass


class TestRunningGuard:
    def _start_running_bracket(self, setup):
        root = setup["working_directory_root"]
        working_directory = os.path.join(root, "T_300.00_1000.00")
        os.makedirs(working_directory)
        open(os.path.join(working_directory, "log.lammps"), "w").close()  # fresh activity
        open(single_module._marker_path(root, "started", (300.0, 1000.0)), "w").close()
        from phase_diagram_workflows.free_energies.ts_convergence.single import _record_bracket

        _record_bracket(root, (300.0, 1000.0))

    def test_bracket_with_recent_activity_is_left_alone(self, calphy, setup):
        fake = calphy({1000.0: 0.001})
        self._start_running_bracket(setup)

        result = refine_temperature_bracket_with_resubmission(**setup)
        assert result["status"] == "running"
        assert fake.runs == []

    def test_force_takes_it_over_anyway(self, calphy, setup):
        fake = calphy({1000.0: 0.001})
        self._start_running_bracket(setup)

        result = refine_temperature_bracket_with_resubmission(**setup, force=True)
        assert result["status"] == "converged"
        assert len(fake.runs) == 1

    def test_silent_bracket_is_taken_over_after_the_stale_threshold(self, calphy, setup):
        fake = calphy({1000.0: 0.001})
        self._start_running_bracket(setup)
        old = time.time() - 7 * 3600
        for path in [os.path.join(setup["working_directory_root"], "T_300.00_1000.00", "log.lammps"),
                     single_module._marker_path(setup["working_directory_root"], "started", (300.0, 1000.0))]:
            os.utime(path, (old, old))
        os.utime(os.path.join(setup["working_directory_root"], "T_300.00_1000.00"), (old, old))

        result = refine_temperature_bracket_with_resubmission(**setup)
        assert result["status"] == "converged"
        assert len(fake.runs) == 1

    def test_crashed_calphy_releases_the_start_marker_so_a_rerun_is_not_blocked(self, monkeypatch, setup):
        def crash(**kwargs):
            raise RuntimeError("LAMMPS died")

        monkeypatch.setattr(single_module, "calc_free_energy_with_calphy", crash)
        with pytest.raises(RuntimeError, match="LAMMPS died"):
            refine_temperature_bracket_with_resubmission(**setup)
        assert not os.path.exists(single_module._marker_path(setup["working_directory_root"], "started", (300.0, 1000.0)))


class TestConfigurationPinning:
    def test_same_configuration_may_be_rerun_with_different_tolerance_and_steps(self, calphy, setup):
        calphy({1000.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        refine_temperature_bracket_with_resubmission(**{**setup, "tolerance": 0.5, "step_upper": 50.0})  # decision inputs, not part of the pin

    def test_changed_calphy_parameter_is_refused(self, calphy, setup):
        calphy({1000.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        setup["calphy_parameters"] = {**setup["calphy_parameters"], "n_equilibration_steps": 99}
        with pytest.raises(ValueError, match="different structure, potential or calphy"):
            refine_temperature_bracket_with_resubmission(**setup)

    def test_changed_structure_is_refused(self, calphy, setup):
        calphy({1000.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        setup["input_structure"] = bulk("Al", cubic=True).repeat(2)
        with pytest.raises(ValueError, match="different structure, potential or calphy"):
            refine_temperature_bracket_with_resubmission(**setup)

    def test_changed_potential_is_refused(self, calphy, setup):
        calphy({1000.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        setup["potential_df"] = pd.DataFrame({"Config": [["pair_style other\n", "pair_coeff * *\n"]], "Species": [["Al"]]})
        with pytest.raises(ValueError, match="different structure, potential or calphy"):
            refine_temperature_bracket_with_resubmission(**setup)

    def test_changed_initial_bracket_is_refused(self, calphy, setup):
        calphy({1000.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup)
        setup["initial_bracket"] = (300.0, 900.0)
        with pytest.raises(ValueError, match="different structure, potential or calphy"):
            refine_temperature_bracket_with_resubmission(**setup)


class TestIterationCap:
    """A structure that never converges must not keep queueing jobs: every bracket is a full job."""

    def test_default_cap_is_five(self):
        assert inspect.signature(refine_temperature_bracket_with_resubmission).parameters["max_iterations"].default == 5

    def test_stops_after_max_iterations_narrowing_steps(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))  # never converges
        setup.update(initial_bracket=(100.0, 1000.0), step_upper=10.0, max_iterations=3)
        refine_temperature_bracket_with_resubmission(**setup)

        assert [run[1] for run in fake.runs] == [1000.0, 990.0, 980.0, 970.0]  # first bracket + 3 narrowing steps
        assert RecordingExecutor.results[0]["status"] == "limit_reached"  # the last step's own outcome
        assert len(RecordingExecutor.instances) == 3  # and it submitted nothing after that

    def test_zero_runs_a_single_bracket_and_submits_nothing(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        setup.update(initial_bracket=(100.0, 1000.0), step_upper=10.0, max_iterations=0)
        result = refine_temperature_bracket_with_resubmission(**setup)

        assert result["status"] == "limit_reached"
        assert len(fake.runs) == 1
        assert RecordingExecutor.instances == []

    def test_default_cap_applies_when_not_given(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        setup.update(initial_bracket=(100.0, 1000.0), step_upper=10.0)
        refine_temperature_bracket_with_resubmission(**setup)
        assert len(fake.runs) == 6  # first bracket + 5 narrowing steps

    def test_rerun_with_a_larger_cap_continues_from_where_it_stopped(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        setup.update(initial_bracket=(100.0, 1000.0), step_upper=10.0)
        refine_temperature_bracket_with_resubmission(**setup, max_iterations=1)
        assert [run[1] for run in fake.runs] == [1000.0, 990.0]

        refine_temperature_bracket_with_resubmission(**setup, max_iterations=3)
        assert [run[1] for run in fake.runs] == [1000.0, 990.0, 980.0, 970.0]  # earlier brackets not run again

    def test_convergence_before_the_cap_is_unaffected(self, calphy, setup):
        fake = calphy({1000.0: 0.03, 900.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup, max_iterations=1)
        assert RecordingExecutor.results[0]["status"] == "converged"
        assert len(fake.runs) == 2

    @pytest.mark.parametrize("bad", [-1, 2.5, True, "3"])
    def test_anything_but_a_non_negative_integer_or_none_is_refused(self, setup, bad):
        with pytest.raises(ValueError, match="max_iterations"):
            refine_temperature_bracket_with_resubmission(**setup, max_iterations=bad)

    def test_none_means_until_converged_and_warns(self, calphy, setup):
        fake = calphy({1000.0: 0.03, 900.0: 0.02, 800.0: 0.01, 700.0: 0.02, 600.0: 0.02, 500.0: 0.02, 400.0: 0.001})
        with pytest.warns(UserWarning, match="max_iterations=None"):
            refine_temperature_bracket_with_resubmission(**setup, max_iterations=None)

        assert [run[1] for run in fake.runs] == [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0]  # past the default cap of 5
        assert RecordingExecutor.results[0]["status"] == "converged"

    def test_none_still_stops_when_the_bracket_cannot_be_narrowed_further(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        setup.update(initial_bracket=(700.0, 720.0), step_upper=10.0)
        with pytest.warns(UserWarning, match="max_iterations=None"):
            refine_temperature_bracket_with_resubmission(**setup, max_iterations=None)

        assert [run[1] for run in fake.runs] == [720.0, 710.0]
        assert RecordingExecutor.results[0]["status"] == "exhausted"

    def test_finite_cap_does_not_warn(self, calphy, setup, recwarn):
        calphy({1000.0: 0.001})
        refine_temperature_bracket_with_resubmission(**setup, max_iterations=3)
        assert not [w for w in recwarn if "max_iterations" in str(w.message)]
