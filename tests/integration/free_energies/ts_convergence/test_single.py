"""
Integration tests for phase_diagram_workflows.free_energies.ts_convergence.single.

Every test here exercises the real forward/backward TI output format, real
executor timing, and real disk state -- not a guessed approximation of any of
them. But a real calphy calculation is genuinely expensive (real LAMMPS MD,
even at ~100 equilibration/switching steps), and most of what this file
tests is bracket *bookkeeping* (pending/converged/resubmitted decisions,
narrowing, chaining) rather than physics: the same real result is a valid
stand-in for "a real result at this bracket" regardless of which bracket's
directory it's read back from, since none of these tests assert anything
about the specific numeric criterion value, only its comparison against
generous/deterministically-unreachable tolerances (see `ts_overlap_criterion`
-- always >= 0, so a real result is always "converged" under a generous
tolerance and never "converged" under a negative one).

So only a handful of tests -- the ones testing real async executor timing
(genuine SingleNodeExecutor, a real background thread/subprocess) or
submit_bracket's own real wiring -- run calphy themselves. Everything else
reuses the single real result computed once by `real_ts_result_source`,
either via `StubResultExecutor` (for code paths where the executor is handed
the real function directly) or via `_patch_real_calc` (for the
self-resubmitting chain, where the real function is called by name from
inside the submitted task, not passed through the executor).
"""

import shutil
from concurrent.futures import Future
from unittest.mock import patch

import pytest
from ase.build import bulk
from executorlib import SingleNodeExecutor
from lammpsparser import get_potential_by_name

import phase_diagram_workflows.free_energies.ts_convergence.single as single_module
from phase_diagram_workflows.free_energies.ti_calculator import (
    calc_free_energy_with_calphy,
    gather_calphy_results_detailed,
)
from phase_diagram_workflows.free_energies.ts_convergence.single import (
    _bracket_working_directory,
    _find_tried_brackets,
    refine_temperature_bracket_manually,
    submit_bracket,
    submit_bracket_chain,
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


class DependencyResolvingEagerExecutor(EagerExecutor):
    """Like EagerExecutor, but also resolves Future-valued kwargs before
    calling fn, emulating (synchronously) the DependencyTaskScheduler
    behavior that SingleNodeExecutor/SlurmClusterExecutor/FluxClusterExecutor
    all provide by default. submit_bracket_chain relies on that resolution
    -- each step's `previous_result` kwarg is the raw Future from the step
    before it -- so a plain EagerExecutor (which just calls fn(**kwargs)
    with whatever it's given, Future or not) isn't a valid stand-in here.
    """

    def submit(self, fn, **kwargs):
        resolved_kwargs = {
            key: (value.result() if isinstance(value, Future) else value) for key, value in kwargs.items()
        }
        return super().submit(fn, **resolved_kwargs)


class _NeverDoneExecutor:
    """Submits nothing real: returns a Future that never completes.

    Demonstrates that "pending" is entirely a disk-state question:
    refine_temperature_bracket_manually never inspects the executor's
    Future at all (not even `.done()`), so an executor this deliberately
    unhelpful still produces the exact same "pending" result as a real one
    -- what matters is only whether a result exists on disk yet.
    """

    def submit(self, fn, **kwargs):
        return Future()


class StubResultExecutor:
    """Like EagerExecutor, but materializes a pre-computed real result
    instead of rerunning calphy.

    Only valid for code paths (submit_bracket, refine_temperature_bracket_manually)
    that always hand `calc_free_energy_with_calphy` itself to
    `executor.submit` as `fn` -- this ignores `fn` entirely and copies
    `source_directory` (a real, already-computed calphy output) to
    `kwargs["working_directory"]` instead. The copied files are genuine
    calphy output, still parsed by the real `gather_calphy_results_detailed`,
    so every downstream assertion about forward/backward TI data sees the
    real format -- only the ~15s of LAMMPS MD itself is skipped, which is
    fine for tests asserting bracket bookkeeping (pending/converged/
    resubmitted), not physics.
    """

    def __init__(self, source_directory):
        self.source_directory = source_directory
        self.submit_calls = []

    def submit(self, fn, **kwargs):
        self.submit_calls.append(kwargs)
        working_directory = kwargs["working_directory"]
        future = Future()
        try:
            shutil.copytree(self.source_directory, working_directory)
            df = gather_calphy_results_detailed(working_directory)
            future.set_result((None, df))
        except Exception as exc:  # pragma: no cover - surfaced via future.result()
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _fake_calc_free_energy_with_calphy(source_directory):
    """Build a drop-in replacement for calc_free_energy_with_calphy that
    materializes `source_directory` (a real, already-computed calphy run)
    instead of running calphy again.

    Used to patch `single_module.calc_free_energy_with_calphy` for the
    self-resubmitting chain, where the real function is called by name from
    *inside* the submitted task (`_calc_and_maybe_resubmit`), not handed to
    the executor -- so a stub executor (as used for the other entry points)
    can't intercept it; the module-level name has to be patched instead.
    """

    def _fake(*, input_structure, potential_df, calphy_parameters, working_directory):
        shutil.copytree(source_directory, working_directory)
        return None, gather_calphy_results_detailed(working_directory)

    return _fake


@pytest.fixture(scope="module")
def potential_df():
    pot = get_potential_by_name("1999--Mishin-Y--Al--LAMMPS--ipr1")
    return pot.to_frame().transpose()


@pytest.fixture(scope="module")
def small_structure():
    return bulk("Al", cubic=True).repeat(2)  # 32 atoms: small and fast


@pytest.fixture(scope="module")
def real_ts_result_source(small_structure, potential_df, tmp_path_factory):
    """A single real, small, fast ts-mode calphy run, computed once and
    reused (read-only, via copies) as the "real result" for every bracket in
    every test that only cares about bracket bookkeeping rather than
    physics. See module docstring for why reusing one real result across
    differently-named brackets is valid here."""
    working_directory = str(tmp_path_factory.mktemp("real_ts_result_source"))
    calc_free_energy_with_calphy(
        input_structure=small_structure,
        potential_df=potential_df,
        calphy_parameters={
            "mode": "ts",
            "temperature": [700.0, 720.0],
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
        },
        working_directory=working_directory,
    )
    return working_directory


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
        executor = EagerExecutor()

        future, working_directory = submit_bracket(
            input_structure=small_structure,
            bracket=(700.0, 720.0),
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=executor,
            working_directory_root=str(tmp_path),
        )

        assert working_directory == _bracket_working_directory(str(tmp_path), 700.0, 720.0)
        assert future.done()
        _, df = future.result()
        assert bool(df.iloc[0]["status"])
        assert executor.submit_calls[0]["calphy_parameters"]["temperature"] == [700.0, 720.0]


class TestRefineTemperatureBracketReal:
    def test_first_call_always_submits_and_reports_pending(
        self, small_structure, potential_df, base_ts_params, tmp_path, real_ts_result_source
    ):
        # Nothing on disk yet, so the very first call can only submit
        # fire-and-forget and report "pending" -- regardless of how fast
        # the executor happens to run the work underneath it, since this
        # function never checks the executor's Future at all.
        executor = StubResultExecutor(real_ts_result_source)

        result = refine_temperature_bracket_manually(
            input_structure=small_structure,
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

    def test_converges_on_the_call_after_submission(
        self, small_structure, potential_df, base_ts_params, tmp_path, real_ts_result_source
    ):
        executor = StubResultExecutor(real_ts_result_source)

        refine_temperature_bracket_manually(
            input_structure=small_structure, calphy_parameters=base_ts_params, potential_df=potential_df,
            executor=executor, working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0), tolerance=1.0, step_upper=10.0,
        )
        # By now the (eagerly-executed) real result is on disk; a second
        # call reads it directly rather than touching the executor again.
        result = refine_temperature_bracket_manually(
            input_structure=small_structure, calphy_parameters=base_ts_params, potential_df=potential_df,
            executor=executor, working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0), tolerance=1.0, step_upper=10.0,
        )

        assert result["status"] == "converged"
        assert result["bracket"] == (700.0, 720.0)

    def test_resubmits_then_resolves_narrowed_bracket_on_next_call(
        self, small_structure, potential_df, base_ts_params, tmp_path, real_ts_result_source
    ):
        executor = StubResultExecutor(real_ts_result_source)
        tolerance = -1.0  # deterministically unreachable (criterion is abs(...), always >= 0)

        first = refine_temperature_bracket_manually(
            input_structure=small_structure, calphy_parameters=base_ts_params, potential_df=potential_df,
            executor=executor, working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0), tolerance=tolerance, step_upper=10.0,
        )
        assert first["status"] == "pending"
        assert first["bracket"] == (700.0, 720.0)

        # (700, 720) is now on disk; this call reads it, doesn't converge,
        # and fires off the narrower bracket.
        second = refine_temperature_bracket_manually(
            input_structure=small_structure, calphy_parameters=base_ts_params, potential_df=potential_df,
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
        third_executor = StubResultExecutor(real_ts_result_source)
        third = refine_temperature_bracket_manually(
            input_structure=small_structure,
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
        result = refine_temperature_bracket_manually(
            input_structure=small_structure,
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


class TestSubmitBracketChain:
    """submit_bracket_chain precomputes the full candidate bracket sequence
    and submits every step at once, each depending on the previous step's
    Future -- as opposed to `refine_temperature_bracket_manually` (which
    expects to be called again by hand) or the old recursive
    self-resubmission design (which spun up a fresh executor from inside
    each running job; see git history for why that was replaced).

    Every test here is about that chaining/reuse *bookkeeping*, not physics,
    so `calc_free_energy_with_calphy` -- which `_run_bracket_chain_step`
    calls directly by name rather than receiving from the executor, so
    `StubResultExecutor` can't intercept it -- is patched at the module level
    for the whole class (see `_fast_physics` below). Real async executor
    timing (including genuine dependency resolution across several hops) is
    covered separately, without this patch, in
    `TestSubmitBracketChainRealExecutor`.
    """

    @pytest.fixture(autouse=True)
    def _fast_physics(self, real_ts_result_source, monkeypatch):
        monkeypatch.setattr(
            single_module,
            "calc_free_energy_with_calphy",
            _fake_calc_free_energy_with_calphy(real_ts_result_source),
        )

    def test_converges_on_first_bracket_rest_pass_through(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        working_directory_root = str(tmp_path)

        futures, working_directory = submit_bracket_chain(
            input_structure=small_structure,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=DependencyResolvingEagerExecutor(),
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # generous: converges immediately
            step_upper=10.0,
            max_iterations=3,
        )

        assert len(futures) >= 2  # more than one bracket was precomputed
        first = futures[0].result()
        assert first["status"] == "converged"
        # Every later step is a pure pass-through of the first result --
        # nothing further was computed.
        for future in futures[1:]:
            assert future.result() == first

        tried = _find_tried_brackets(working_directory_root)
        assert tried == [(700.0, 720.0)]  # nothing beyond the first bracket ever ran

    def test_gives_up_after_max_iterations_without_raising(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        working_directory_root = str(tmp_path)

        # step_upper=5.0 keeps every step in this chain well clear of a
        # collapse (700 -> 715 -> 710, never reaching a zero-width bracket),
        # so this test is purely about the max_iterations bound, not the
        # collapse behavior covered separately below.
        futures, _ = submit_bracket_chain(
            input_structure=small_structure,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=DependencyResolvingEagerExecutor(),
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=-1.0,  # deterministically unreachable
            step_upper=5.0,
            max_iterations=2,
        )

        assert len(futures) == 3  # initial bracket + 2 narrowing steps
        assert [f.result()["bracket"] for f in futures] == [(700.0, 720.0), (700.0, 715.0), (700.0, 710.0)]
        assert all(f.result()["status"] == "narrowed" for f in futures)  # never converged, gave up cleanly

        tried = sorted(_find_tried_brackets(working_directory_root))
        assert tried == [(700.0, 710.0), (700.0, 715.0), (700.0, 720.0)]  # all 3 actually ran

    def test_bracket_collapse_ends_the_chain_without_raising(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        # Regression test for a real bug: narrowing (700, 710) by
        # step_upper=10 collapses to a zero-width (700, 700) bracket, which
        # base.step_bracket correctly rejects. In the old recursive
        # self-resubmission design, that ValueError was raised *inside* an
        # already-running, unobserved job and silently vanished -- the chain
        # just stopped one hop early with no error anywhere. Here the whole
        # sequence is decided up front, so the collapse is caught while
        # building the chain (see _precompute_bracket_chain), not while
        # running it.
        working_directory_root = str(tmp_path)

        futures, _ = submit_bracket_chain(
            input_structure=small_structure,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=DependencyResolvingEagerExecutor(),
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=-1.0,  # deterministically unreachable -- would keep narrowing forever otherwise
            step_upper=10.0,
            max_iterations=5,  # much higher than the chain can actually reach before collapsing
        )

        # (700, 720) -> (700, 710) -> collapse: the chain stops itself at 2
        # brackets even though max_iterations=5 would allow more.
        assert len(futures) == 2
        for future in futures:
            future.result()  # must not raise

        tried = sorted(_find_tried_brackets(working_directory_root))
        assert tried == [(700.0, 710.0), (700.0, 720.0)]

    def test_second_independent_bootstrap_reuses_disk_result_without_recomputing(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        # Regression test for a real bug: a second, independent call to
        # submit_bracket_chain for the same structure/bracket (e.g. a
        # notebook cell rerun bootstrapping to whatever's already on disk)
        # must reuse the existing result rather than rerunning calphy.
        # Rerunning is not just wasteful -- calphy's MD is stochastic, so
        # recomputing an already-converged bracket can silently produce a
        # *different*, unconverged criterion the second time, which is
        # exactly how a bracket that already converged ends up narrowed
        # past: the narrowing decision was made from an earlier run's
        # result before a later, redundant one overwrote it.
        working_directory_root = str(tmp_path)

        first_futures, _ = submit_bracket_chain(
            input_structure=small_structure,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=DependencyResolvingEagerExecutor(),
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,  # generous: converges immediately
            step_upper=10.0,
            max_iterations=3,
        )
        first_futures[0].result()

        with patch(
            "phase_diagram_workflows.free_energies.ts_convergence.single.calc_free_energy_with_calphy",
            wraps=single_module.calc_free_energy_with_calphy,
        ) as spy:
            second_futures, _ = submit_bracket_chain(
                input_structure=small_structure,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=DependencyResolvingEagerExecutor(),
                working_directory_root=working_directory_root,
                initial_bracket=(700.0, 720.0),
                tolerance=1.0,
                step_upper=10.0,
                max_iterations=3,
            )
            result = second_futures[0].result()

        spy.assert_not_called()
        assert result["status"] == "converged"

        tried = _find_tried_brackets(working_directory_root)
        assert tried == [(700.0, 720.0)]  # still nothing else ever ran

    def test_later_looser_tolerance_recovers_widest_satisfying_bracket(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        # Regression test: narrow all the way down under a strict (here,
        # unreachable) tolerance, then re-invoke with a looser tolerance
        # that every already-tried bracket actually satisfies. Must recover
        # the *widest* one (700, 720) -- not just report whatever's
        # narrowest on disk -- without recomputing anything.
        working_directory_root = str(tmp_path)

        narrowing_futures, _ = submit_bracket_chain(
            input_structure=small_structure,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=DependencyResolvingEagerExecutor(),
            working_directory_root=working_directory_root,
            initial_bracket=(700.0, 720.0),
            tolerance=-1.0,  # deterministically unreachable -- forces all 3 brackets to run
            step_upper=5.0,
            max_iterations=2,
        )
        narrowing_futures[-1].result()
        tried_before = sorted(_find_tried_brackets(working_directory_root))
        assert tried_before == [(700.0, 710.0), (700.0, 715.0), (700.0, 720.0)]

        with patch(
            "phase_diagram_workflows.free_energies.ts_convergence.single.calc_free_energy_with_calphy",
            wraps=single_module.calc_free_energy_with_calphy,
        ) as spy:
            looser_futures, working_directory = submit_bracket_chain(
                input_structure=small_structure,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=DependencyResolvingEagerExecutor(),
                working_directory_root=working_directory_root,
                initial_bracket=(700.0, 720.0),
                tolerance=1.0,  # generous: every already-tried bracket satisfies this
                step_upper=5.0,
                max_iterations=2,
            )

        spy.assert_not_called()  # recovered from the log -- nothing recomputed
        assert len(looser_futures) == 1  # nothing new was ever submitted
        result = looser_futures[0].result()
        assert result["status"] == "converged"
        assert result["bracket"] == (700.0, 720.0)  # the widest satisfying bracket, not the narrowest

        tried_after = sorted(_find_tried_brackets(working_directory_root))
        assert tried_after == tried_before  # nothing new landed on disk


class TestSubmitBracketChainRealExecutor:
    """Kept separate from TestSubmitBracketChain specifically so it's exempt
    from that class's autouse `_fast_physics` patch: this test is about real
    async executor timing and genuine dependency resolution across several
    hops, not chain bookkeeping, so it needs the genuine
    calc_free_energy_with_calphy.
    """

    @pytest.mark.timeout(300)
    def test_real_dependency_chain_resolves_every_hop(self, small_structure, potential_df, base_ts_params, tmp_path):
        # This is the test that would have caught the original bug: the old
        # recursive self-resubmission design passed with a synchronous
        # EagerExecutor but silently dropped a hop against a real, genuinely
        # async SingleNodeExecutor -- nothing exercised that combination.
        # Here, calling .result() on the *last* Future in the chain is only
        # meaningful if executorlib's real DependencyTaskScheduler actually
        # waited for and resolved every intermediate Future, which is
        # exactly the mechanism being proven.
        #
        # Only 2 real calphy runs (max_iterations=1, not the module default
        # of 10) -- that's already enough to prove a dependency was actually
        # resolved across a hop, and every real run here costs real wall
        # time. Even so, this is heavier than pytest.ini's global 120s
        # budget was calibrated for (that number assumes a single real run,
        # not a multi-hop chain), hence the explicit override above rather
        # than competing for room against much lighter tests.
        working_directory_root = str(tmp_path)

        # hostname_localhost=True: without it, executorlib advertises the
        # machine's real hostname (via gethostname()) for the child process
        # to zmq-connect back to instead of localhost. That's fine on a
        # normal workstation, but on an ephemeral CI runner the child can
        # fail to resolve/connect to that hostname and silently keep
        # retrying forever -- zmq doesn't raise, and executorlib's own
        # receive-loop only gives up if the child process actually exits, so
        # the test just hangs indefinitely instead of failing. Since this is
        # always a single machine (never a multi-node HPC allocation), the
        # real hostname was never needed here anyway.
        executor = SingleNodeExecutor(max_cores=1, hostname_localhost=True)
        try:
            futures, working_directory = submit_bracket_chain(
                input_structure=small_structure,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=executor,
                working_directory_root=working_directory_root,
                initial_bracket=(700.0, 720.0),
                tolerance=-1.0,  # deterministically unreachable -- forces every hop to actually run
                step_upper=5.0,  # stays clear of a collapse for this chain length
                max_iterations=1,
            )
            assert len(futures) == 2  # (700,720) -> (700,715); geometry, not convergence, ends it here
            last_result = futures[-1].result()
        finally:
            executor.shutdown(wait=False, cancel_futures=False)

        assert last_result["status"] == "narrowed"
        assert last_result["bracket"] == (700.0, 715.0)
        tried = sorted(_find_tried_brackets(working_directory_root))
        assert tried == [(700.0, 715.0), (700.0, 720.0)]  # every hop genuinely ran
