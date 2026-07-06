"""
Integration tests for phase_diagram_workflows.free_energies.ts_convergence.ti_binary.

Every test here runs a real, small, fast calphy calculation (Al, EAM
potential via lammpsparser's iprpy database, ~100 equilibration/switching
steps) instead of fabricating result files -- so they exercise the real
forward/backward TI output format, real executor timing, and real disk
state, not a guessed approximation of any of them.
"""

import matplotlib
matplotlib.use("Agg")

import math
import os
import tempfile
from concurrent.futures import Future
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from executorlib import SingleNodeExecutor
from lammpsparser import get_potential_by_name

import phase_diagram_workflows.free_energies.ts_convergence.ti_binary as ti_binary_module
from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed
from phase_diagram_workflows.free_energies.ts_convergence.ti_binary import (
    _bracket_prefix,
    _bracket_working_directory,
    _find_tried_brackets,
    auto_refine_temperature_brackets,
    build_criterion_table,
    get_unique_criterion_table,
    plot_criterion_by_bracket,
    plot_criterion_vs_concentration,
    refine_temperature_bracket,
    submit_bracket,
    submit_self_resubmitting_bracket,
)


class EagerExecutor:
    """Executes the submitted function immediately, in-process.

    Not a mock: `submit` genuinely calls the real function (e.g.
    calc_free_energy_with_calphy) right away and wraps its real result in a
    real, already-completed concurrent.futures.Future. Only the scheduling is
    fake; the calphy execution behind it is real.
    """

    def __init__(self):
        self.submit_calls = []

    def submit(self, fn, **kwargs):
        self.submit_calls.append(kwargs)
        future = Future()
        try:
            future.set_result(fn(**kwargs))
        except Exception as exc:  # pragma: no cover - surfaced via future.result()
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class _NeverDoneExecutor:
    """Submits nothing real: returns a Future that never completes.

    Demonstrates that "pending" is entirely a disk-state question:
    refine_temperature_bracket never inspects the executor's Future at all
    (not even `.done()`), so an executor this deliberately unhelpful still
    produces the exact same "pending" result as a real one -- what matters
    is only whether a result exists on disk yet.
    """

    def submit(self, fn, **kwargs):
        return Future()


def _make_row(atoms, c_in=0.1, phase_type="fcc", reference_phase="solid"):
    return pd.Series({
        "main_element": "Al", "mixing_element": "Mg",
        "phase_type": phase_type, "reference_phase": reference_phase,
        "c_in": c_in, "atoms": atoms,
    })


@pytest.fixture(scope="module")
def potential_df():
    pot = get_potential_by_name("1999--Mishin-Y--Al--LAMMPS--ipr1")
    return pot.to_frame().transpose()


@pytest.fixture(scope="module")
def small_structure():
    return bulk("Al", cubic=True).repeat(2)  # 32 atoms: small and fast


@pytest.fixture
def base_ts_params():
    return {
        "mode": "ts",
        "pressure": 0,
        "n_equilibration_steps": 100,
        "n_switching_steps": 100,
        "n_print_steps": 25,
        "equilibration_control": "berendsen",
        "md": {"thermostat_damping": 0.5},
        "tolerance": {"spring_constant": 0.01, "pressure": 0.5},
        "queue": {"cores": 1, "scheduler": "local"},
        "reference_phase": "solid",
        "file_format": "lammps-data",
    }


class TestSubmitBracket:
    """submit_bracket takes no tolerance parameter at all -- confirmed here
    by never passing one, unlike every other test in this file."""

    def test_submits_real_calphy_run_and_writes_to_expected_directory(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        row = _make_row(small_structure, c_in=0.15)
        executor = EagerExecutor()

        future, working_directory = submit_bracket(
            row=row,
            bracket=(700.0, 720.0),
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=executor,
            working_directory_root=str(tmp_path),
        )

        assert working_directory == _bracket_working_directory(
            str(tmp_path), _bracket_prefix(row), 700.0, 720.0
        )
        assert future.done()
        _, df = future.result()
        assert bool(df.iloc[0]["status"])
        assert executor.submit_calls[0]["calphy_parameters"]["temperature"] == [700.0, 720.0]


class TestRefineTemperatureBracketReal:
    def test_first_call_always_submits_and_reports_pending(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        # Nothing on disk yet, so the very first call can only submit
        # fire-and-forget and report "pending" -- regardless of how fast
        # the executor happens to run the work underneath it, since this
        # function never checks the executor's Future at all.
        row = _make_row(small_structure, c_in=0.1)
        executor = EagerExecutor()

        result = refine_temperature_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=executor,
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,
            step_upper=10.0,
        )

        assert result["status"] == "pending"
        assert result["bracket"] == (700.0, 720.0)
        assert len(executor.submit_calls) == 1

    def test_converges_on_the_call_after_submission(self, small_structure, potential_df, base_ts_params, tmp_path):
        row = _make_row(small_structure, c_in=0.1)
        executor = EagerExecutor()

        refine_temperature_bracket(
            row=row, calphy_parameters=base_ts_params, potential_df=potential_df,
            executor=executor, working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0), tolerance=1.0, step_upper=10.0,
        )
        # By now the (eagerly-executed) real result is on disk; a second
        # call reads it directly rather than touching the executor again.
        result = refine_temperature_bracket(
            row=row, calphy_parameters=base_ts_params, potential_df=potential_df,
            executor=executor, working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0), tolerance=1.0, step_upper=10.0,
        )

        assert result["status"] == "converged"
        assert result["bracket"] == (700.0, 720.0)

    def test_resubmits_then_resolves_narrowed_bracket_on_next_call(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        row = _make_row(small_structure, c_in=0.2)
        executor = EagerExecutor()
        tolerance = -1.0  # deterministically unreachable (criterion is abs(...), always >= 0)

        first = refine_temperature_bracket(
            row=row, calphy_parameters=base_ts_params, potential_df=potential_df,
            executor=executor, working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0), tolerance=tolerance, step_upper=10.0,
        )
        assert first["status"] == "pending"
        assert first["bracket"] == (700.0, 720.0)

        # (700, 720) is now on disk; this call reads it, doesn't converge,
        # and fires off the narrower bracket.
        second = refine_temperature_bracket(
            row=row, calphy_parameters=base_ts_params, potential_df=potential_df,
            executor=executor, working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0), tolerance=tolerance, step_upper=10.0,
        )
        assert second["status"] == "resubmitted"
        assert second["bracket"] == (700.0, 710.0)

        # Fresh call (simulating a new session, generous tolerance now):
        # (700, 720) is also on disk and trivially satisfies this generous
        # tolerance too -- and it's wider (cheaper, less-narrowed) than
        # (700, 710), so it should be preferred over just resolving to the
        # narrowest bracket tried so far. Only narrowing under the earlier,
        # unreachable tolerance ever justified going past (700, 720) in the
        # first place; there's no reason to keep treating that as settled
        # once a looser tolerance is in effect.
        third_executor = EagerExecutor()
        third = refine_temperature_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=third_executor,
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,
            step_upper=10.0,
        )
        assert third["bracket"] == (700.0, 720.0)
        assert third["status"] == "converged"
        assert len(third_executor.submit_calls) == 0  # recovered from disk, nothing (re)submitted

    def test_pending_regardless_of_what_the_executor_does(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        # Nothing on disk yet -- "pending" holds even for an executor that
        # will never actually finish, since this function never inspects
        # the Future it gets back.
        row = _make_row(small_structure, c_in=0.3)

        result = refine_temperature_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=_NeverDoneExecutor(),
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,
            step_upper=10.0,
        )
        assert result["status"] == "pending"
        assert result["bracket"] == (700.0, 720.0)


class TestAutoRefineTemperatureBrackets:
    def test_all_rows_converge_in_first_round(self, small_structure, potential_df, base_ts_params, tmp_path):
        structures_df = pd.DataFrame([
            _make_row(small_structure, c_in=0.1),
            _make_row(small_structure, c_in=0.2),
        ])

        result_df = auto_refine_temperature_brackets(
            structures_df=structures_df,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # generous: converges immediately
            step_upper=10.0,
            max_iterations=3,
        )

        assert list(result_df.index) == list(structures_df.index)
        assert (result_df["status"] == "converged").all()
        assert (result_df["bracket"] == (700.0, 720.0)).all()

    def test_gives_up_after_max_iterations_without_raising(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        structures_df = pd.DataFrame([_make_row(small_structure, c_in=0.3)])

        result_df = auto_refine_temperature_brackets(
            structures_df=structures_df,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 740.0),
            tolerance=1e-12,  # unreachable: never converges
            step_upper=10.0,
            max_iterations=2,
        )

        row = result_df.iloc[0]
        assert row["status"] == "resubmitted"  # gave up, didn't raise
        # max_iterations=2 narrowing attempts for this row: (700,740)->(700,730)
        # on the first, (700,730)->(700,720) on the second, then it stops.
        assert row["bracket"] == (700.0, 720.0)

    def test_independent_rows_progress_concurrently(self, small_structure, potential_df, base_ts_params, tmp_path):
        # One row converges immediately; the other needs one narrowing step.
        # Both should be driven in the same call, not one-at-a-time.
        structures_df = pd.DataFrame([
            _make_row(small_structure, c_in=0.4),
            _make_row(small_structure, c_in=0.5),
        ])

        calls = []

        class RecordingEagerExecutor(EagerExecutor):
            def submit(self, fn, **kwargs):
                calls.append(kwargs["working_directory"])
                return super().submit(fn, **kwargs)

        result_df = auto_refine_temperature_brackets(
            structures_df=structures_df,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=RecordingEagerExecutor(),
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,
            step_upper=10.0,
            max_iterations=3,
        )

        assert (result_df["status"] == "converged").all()
        # Both rows' initial brackets were submitted in the same (first) round.
        assert sum("0.400000" in c for c in calls) >= 1
        assert sum("0.500000" in c for c in calls) >= 1


class TestAutoRefineTemperatureBracketsRealExecutor:
    """Uses executorlib's own SingleNodeExecutor -- a genuine async executor
    with a real background thread and real on-disk caching -- instead of the
    synchronous EagerExecutor stand-in used everywhere else in this file.

    EagerExecutor runs the real calphy calculation, but *synchronously*
    inside submit(), so it can never exercise the "submitted now, resolves
    later via a background thread" timing that a real executorlib executor
    has. That gap is exactly what let the "checking .done() immediately"
    bug slip through untested before. This test closes it.
    """

    def test_converges_with_real_async_executor(self, small_structure, potential_df, base_ts_params, tmp_path):
        structures_df = pd.DataFrame([_make_row(small_structure, c_in=0.6)])
        cache_directory = str(tmp_path / "cache")
        working_directory_root = str(tmp_path / "work")

        executor = SingleNodeExecutor(cache_directory=cache_directory, max_cores=1)
        try:
            result_df = auto_refine_temperature_brackets(
                structures_df=structures_df,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=executor,
                working_directory_root=working_directory_root,
                initial_bracket=(700.0, 720.0),
                tolerance=1.0,  # generous: converges on the first real result
                step_upper=10.0,
                # Kept low: calphy on a tiny 32-atom cell can be numerically
                # unstable and may never actually converge, so this caps
                # worst-case wall time regardless.
                max_iterations=3,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=False)

        row = result_df.iloc[0]
        assert row["status"] == "converged"
        assert row["bracket"] == (700.0, 720.0)

    def test_narrows_with_real_async_executor(self, small_structure, potential_df, base_ts_params, tmp_path):
        structures_df = pd.DataFrame([_make_row(small_structure, c_in=0.7)])
        cache_directory = str(tmp_path / "cache")
        working_directory_root = str(tmp_path / "work")

        executor = SingleNodeExecutor(cache_directory=cache_directory, max_cores=1)
        try:
            result_df = auto_refine_temperature_brackets(
                structures_df=structures_df,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=executor,
                working_directory_root=working_directory_root,
                initial_bracket=(700.0, 720.0),
                tolerance=-1.0,  # deterministically unreachable
                step_upper=10.0,
                max_iterations=1,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=False)

        row = result_df.iloc[0]
        assert row["status"] == "resubmitted"
        assert row["bracket"] == (700.0, 710.0)


class TestSubmitSelfResubmittingBracket:
    """submit_self_resubmitting_bracket submits once and returns right away;
    any further resubmission happens *inside* the submitted task itself
    (`_calc_and_maybe_resubmit`), not in the calling process -- this is the
    "submit and walk away" mode, as opposed to auto_refine_temperature_brackets
    (which blocks in the caller) or refine_temperature_bracket (which expects
    to be called again by hand).
    """

    def test_converges_without_chaining(self, small_structure, potential_df, base_ts_params, tmp_path):
        row = _make_row(small_structure, c_in=0.8)
        working_directory_root = str(tmp_path)

        future, working_directory = submit_self_resubmitting_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            executor_factory=EagerExecutor,
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # generous: converges immediately
            step_upper=10.0,
            # Kept low: calphy on a tiny 32-atom cell can be numerically
            # unstable and may never actually converge, so this caps
            # worst-case wall time regardless.
            max_iterations=3,
        )

        _, df = future.result()
        assert df.iloc[0]["forward_energy_diff"] is not None

        prefix = _bracket_prefix(row)
        tried = _find_tried_brackets(working_directory_root, prefix)
        assert tried == [(700.0, 720.0)]  # nothing chained -- converged on the first try

    def test_second_independent_bootstrap_reuses_disk_result_without_recomputing(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        # Regression test for a real bug: a second, independent call to
        # submit_self_resubmitting_bracket for the same row/bracket (e.g. a
        # notebook cell rerun bootstrapping to whatever's already on disk)
        # must reuse the existing result rather than rerunning calphy.
        # Rerunning is not just wasteful -- calphy's MD is stochastic, so
        # recomputing an already-converged bracket can silently produce a
        # *different*, unconverged criterion the second time, which is
        # exactly how a bracket that already converged ends up narrowed
        # past: the narrowing decision was made from an earlier run's
        # result before a later, redundant one overwrote it.
        row = _make_row(small_structure, c_in=1.1)
        working_directory_root = str(tmp_path)

        future, _ = submit_self_resubmitting_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            executor_factory=EagerExecutor,
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # generous: converges immediately
            step_upper=10.0,
            max_iterations=3,
        )
        future.result()

        with patch(
            "phase_diagram_workflows.free_energies.ts_convergence.ti_binary.calc_free_energy_with_calphy",
            wraps=ti_binary_module.calc_free_energy_with_calphy,
        ) as spy:
            second_future, _ = submit_self_resubmitting_bracket(
                row=row,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=EagerExecutor(),
                executor_factory=EagerExecutor,
                working_directory_root=working_directory_root,
                initial_bracket=(700.0, 720.0),
                tolerance=1.0,
                step_upper=10.0,
                max_iterations=3,
            )
            _, df = second_future.result()

        spy.assert_not_called()
        assert df.iloc[0]["forward_energy_diff"] is not None

        prefix = _bracket_prefix(row)
        tried = _find_tried_brackets(working_directory_root, prefix)
        assert tried == [(700.0, 720.0)]  # still nothing chained

    def test_does_not_chain_past_a_wider_already_converged_bracket(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        # Regression test for the same bug from the other direction: seed
        # history with a wide, already-converged bracket plus a narrower one
        # that also exists on disk (as if narrowing had previously continued
        # under a stricter tolerance). Bootstrapping fresh with today's
        # (looser) tolerance must recover the wider bracket and stop --
        # not keep resubmitting narrower ones just because a narrower
        # bracket happens to be what's currently on disk.
        row = _make_row(small_structure, c_in=1.2)
        working_directory_root = str(tmp_path)

        wide_future, _ = submit_self_resubmitting_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            executor_factory=EagerExecutor,
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # generous: converges immediately at (700, 720)
            step_upper=10.0,
            max_iterations=3,
        )
        wide_future.result()

        # A narrower bracket also exists on disk, as if produced by an
        # earlier, stricter tolerance.
        submit_bracket(row, (700.0, 710.0), base_ts_params, potential_df, EagerExecutor(), working_directory_root)

        prefix = _bracket_prefix(row)
        tried_before = sorted(_find_tried_brackets(working_directory_root, prefix))
        assert tried_before == [(700.0, 710.0), (700.0, 720.0)]

        second_future, _ = submit_self_resubmitting_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            executor_factory=EagerExecutor,
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # same, generous tolerance -- both brackets satisfy it
            step_upper=10.0,
            max_iterations=3,
        )
        _, resolved_df = second_future.result()

        # Must recover the wider (700, 720) bracket and stop there -- not
        # keep narrowing past it just because (700, 710) is what's narrowest
        # on disk. No new bracket should have been submitted at all.
        tried_after = sorted(_find_tried_brackets(working_directory_root, prefix))
        assert tried_after == tried_before

        wide_working_directory = _bracket_working_directory(working_directory_root, prefix, 700.0, 720.0)
        wide_df = gather_calphy_results_detailed(wide_working_directory)
        assert np.array_equal(
            resolved_df.iloc[0]["forward_energy_diff"][0], wide_df.iloc[0]["forward_energy_diff"][0]
        )

    def test_chains_until_max_iterations_without_raising(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        row = _make_row(small_structure, c_in=0.9)
        working_directory_root = str(tmp_path)

        # EagerExecutor runs fn(**kwargs) synchronously inside submit(), so
        # by the time this call returns, the entire self-resubmitting chain
        # (up to max_iterations) has already run in-process -- confirmed via
        # what actually landed on disk, exercising the real chaining logic
        # rather than a mock of it.
        future, working_directory = submit_self_resubmitting_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            executor_factory=EagerExecutor,
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 740.0),
            tolerance=-1.0,  # deterministically unreachable
            step_upper=10.0,
            max_iterations=2,
        )
        future.result()

        prefix = _bracket_prefix(row)
        tried = sorted(_find_tried_brackets(working_directory_root, prefix))
        assert tried == [(700.0, 720.0), (700.0, 730.0), (700.0, 740.0)]

    def test_converges_with_real_async_executor(self, small_structure, potential_df, base_ts_params, tmp_path):
        # Proves executor_factory genuinely round-trips through cloudpickle
        # to wherever _calc_and_maybe_resubmit actually executes, using a
        # real async executor rather than the synchronous EagerExecutor.
        # max_iterations is kept low and the assertion below doesn't require
        # actual convergence: calphy on a tiny 32-atom cell can be
        # numerically unstable and may never converge, and that's not what
        # this test is checking -- only that the chain runs and stays bounded.
        cache_directory = str(tmp_path / "cache")
        working_directory_root = str(tmp_path / "work")
        row = _make_row(small_structure, c_in=1.0)

        def executor_factory():
            return SingleNodeExecutor(cache_directory=cache_directory, max_cores=1)

        executor = executor_factory()
        try:
            future, working_directory = submit_self_resubmitting_bracket(
                row=row,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=executor,
                executor_factory=executor_factory,
                working_directory_root=working_directory_root,
                initial_bracket=(700.0, 720.0),
                tolerance=1.0,  # generous: converges on the first real result, typically
                step_upper=10.0,
                max_iterations=2,
            )
            _, df = future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=False)

        assert df.iloc[0]["forward_energy_diff"] is not None
        prefix = _bracket_prefix(row)
        tried = _find_tried_brackets(working_directory_root, prefix)
        assert 1 <= len(tried) <= 3  # bounded by max_iterations regardless of convergence


@pytest.fixture(scope="module")
def criterion_sweep(small_structure, potential_df, tmp_path_factory):
    """Runs one real, small sweep: one concentration that converges
    immediately, one that needs a single resubmission. Shared across the
    criterion-table/plotting tests so the real calphy calls happen once.
    """
    working_directory_root = str(tmp_path_factory.mktemp("criterion_sweep"))
    base_params = {
        "mode": "ts",
        "pressure": 0,
        "n_equilibration_steps": 100,
        "n_switching_steps": 100,
        "n_print_steps": 25,
        "equilibration_control": "berendsen",
        "md": {"thermostat_damping": 0.5},
        "tolerance": {"spring_constant": 0.01, "pressure": 0.5},
        "queue": {"cores": 1, "scheduler": "local"},
        "reference_phase": "solid",
        "file_format": "lammps-data",
    }

    structures_df = pd.DataFrame([
        _make_row(small_structure, c_in=0.1),
        _make_row(small_structure, c_in=0.2),
    ])

    # Two calls each: the first only submits (nothing on disk yet, so it can
    # only report "pending"); the second reads the now-available result and
    # decides converged/resubmitted.
    executor0 = EagerExecutor()
    for _ in range(2):
        refine_temperature_bracket(
            row=structures_df.iloc[0],
            calphy_parameters=base_params,
            potential_df=potential_df,
            executor=executor0,
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # converges immediately once checked
            step_upper=10.0,
        )

    executor1 = EagerExecutor()
    for _ in range(2):
        refine_temperature_bracket(
            row=structures_df.iloc[1],
            calphy_parameters=base_params,
            potential_df=potential_df,
            executor=executor1,
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            # Negative: deterministically unreachable (criterion is abs(...), always >= 0).
            tolerance=-1.0,
            step_upper=10.0,
        )

    return structures_df, working_directory_root


class TestCriterionTableAndPlots:
    def test_build_criterion_table(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)

        expected_columns = {
            "main_element", "mixing_element", "phase_type", "reference_phase",
            "c_in", "t_low", "t_high", "criterion", "working_directory",
        }
        assert expected_columns.issubset(table.columns)

        converged_rows = table[table["c_in"] == 0.1]
        resubmitted_rows = table[table["c_in"] == 0.2]
        assert len(converged_rows) == 1
        assert len(resubmitted_rows) == 2
        assert converged_rows["criterion"].notna().all()
        assert resubmitted_rows["criterion"].notna().all()
        assert set(zip(resubmitted_rows["t_low"], resubmitted_rows["t_high"])) == {
            (700.0, 720.0), (700.0, 710.0),
        }

    def test_get_unique_criterion_table_picks_current_bracket(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)
        unique = get_unique_criterion_table(table, initial_bracket=(700.0, 720.0))

        assert len(unique) == 2
        narrowed_row = unique[unique["c_in"] == 0.2].iloc[0]
        assert (narrowed_row["t_low"], narrowed_row["t_high"]) == (700.0, 710.0)

    def test_get_unique_criterion_table_tolerance_prefers_widest_converged_bracket(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)

        # A very loose tolerance is satisfied by every attempted bracket, so
        # the widest (least-narrowed, cheapest) one should be reported
        # instead of falling back to the narrowest bracket on disk -- this is
        # what lets a later, looser tolerance recover an earlier bracket that
        # was already good enough, rather than only ever seeing the extra
        # narrowing that a previous, stricter tolerance forced.
        unique = get_unique_criterion_table(
            table, initial_bracket=(700.0, 720.0), tolerance=math.inf
        )

        widened_row = unique[unique["c_in"] == 0.2].iloc[0]
        assert (widened_row["t_low"], widened_row["t_high"]) == (700.0, 720.0)

    def test_get_unique_criterion_table_tolerance_falls_back_when_unmet(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)

        # Deterministically unreachable, so no attempt satisfies it -- falls
        # back to the narrowest bracket on disk, same as the no-tolerance case.
        unique = get_unique_criterion_table(
            table, initial_bracket=(700.0, 720.0), tolerance=-1.0
        )

        narrowed_row = unique[unique["c_in"] == 0.2].iloc[0]
        assert (narrowed_row["t_low"], narrowed_row["t_high"]) == (700.0, 710.0)

    def test_plot_criterion_vs_concentration(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)
        unique = get_unique_criterion_table(table, initial_bracket=(700.0, 720.0))

        fig, ax = plot_criterion_vs_concentration(unique, tolerance=0.5)
        assert len(ax.collections) == 1  # the scatter
        assert len(ax.lines) == 1  # the tolerance reference line

    def test_plot_criterion_by_bracket(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)

        fig, ax = plot_criterion_by_bracket(table, group_by="t_high", tolerance=0.5)
        # distinct t_high values across the whole table: 720 (both rows) and 710 (row2 only)
        assert len(ax.lines) == 2 + 1  # one line per t_high group, plus the tolerance line
