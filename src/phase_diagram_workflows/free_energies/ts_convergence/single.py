"""Temperature-scaling (TS) bracket convergence for ONE structure.

Narrow the ``(t_low, t_high)`` bracket of a reversible-scaling run until the forward/backward
TI overlap criterion is within tolerance. Three ways to drive the same search, all sharing one
``bracket_log.csv`` per structure:

- ``refine_temperature_bracket_manually``: you call it again by hand; each call checks disk and
  submits the next bracket if needed.
- ``refine_temperature_bracket_with_chain``: submit the whole narrowing sequence up front as an executor
  dependency chain (needs an executor that resolves Future arguments, e.g. a cluster executor).
- ``refine_temperature_bracket_with_resubmission``: submit once and close everything; every bracket
  is its own job and submits the next one itself (see ``ExecutorSpec``).

The two automatic drivers take ``max_iterations`` (default 5): the most narrowing steps beyond the first bracket, since
every bracket is a full job. ``None`` means "until converged" and warns.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import warnings
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from phase_diagram_workflows.utils.executor_spec import ExecutorSpec, _DETACHABLE_EXECUTOR_NAMES  # noqa: F401  (re-exported: ExecutorSpec used to be defined here)
from phase_diagram_workflows.free_energies.ti_calculator import (
    calc_free_energy_with_calphy,
    gather_calphy_results_detailed,
)
from phase_diagram_workflows.free_energies.ts_convergence.base import (
    decide_next_bracket,
    pick_best_converged_bracket,
    resolve_current_bracket,
    scale_steps_to_bracket_width,
    step_bracket,
    ts_overlap_criterion,
)

_BRACKET_MARKER = "T_"

DEFAULT_MAX_ITERATIONS = 5
# `max_iterations=None` means "until converged", but never literally forever: a tiny step size would otherwise
# queue an absurd number of full jobs.
_UNBOUNDED_CEILING = 1000


def _check_max_iterations(max_iterations: Optional[int]) -> None:
    """Validate `max_iterations`; warn loudly when it is None (no cap)."""
    if max_iterations is None:
        warnings.warn(
            "max_iterations=None: the bracket keeps narrowing until it converges or can no longer be narrowed. "
            "Every bracket is a full job, so with a small step size this can queue many of them "
            f"(safety ceiling: {_UNBOUNDED_CEILING} brackets). Prefer a finite max_iterations.",
            UserWarning,
            stacklevel=3,
        )
        return
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError(f"max_iterations must be a non-negative integer, or None for 'until converged'; got {max_iterations!r}.")


def _bracket_working_directory(working_directory_root: str, t_low: float, t_high: float) -> str:
    return os.path.join(working_directory_root, f"{_BRACKET_MARKER}{t_low:.2f}_{t_high:.2f}")


def _find_tried_brackets(working_directory_root: str) -> List[Tuple[float, float]]:
    """Recover every bracket already attempted, by scanning disk.

    Stateless replacement for a job table: the current bracket is always
    recomputed from what's already on disk under `working_directory_root`,
    never stored separately.
    """
    tried: List[Tuple[float, float]] = []
    if not os.path.isdir(working_directory_root):
        return tried

    for name in os.listdir(working_directory_root):
        if not name.startswith(_BRACKET_MARKER):
            continue
        if not os.path.isdir(os.path.join(working_directory_root, name)):
            continue

        parts = name[len(_BRACKET_MARKER):].split("_")
        if len(parts) != 2:
            continue  # unparsable suffix (e.g. a manual backup folder) -- skip, don't crash
        try:
            t_low, t_high = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        tried.append((t_low, t_high))

    return tried


def _tried_bracket_criteria(
    working_directory_root: str,
    tried_brackets: List[Tuple[float, float]],
) -> Dict[Tuple[float, float], float]:
    """Read the TI overlap criterion for every already-tried bracket that has one.

    Brackets with no result yet (still running) or incomplete results (e.g.
    fe mode) are silently omitted rather than reported as some placeholder
    value -- callers should treat "not in this dict" as "unknown", not "bad".
    """
    criteria: Dict[Tuple[float, float], float] = {}
    for t_low, t_high in tried_brackets:
        working_directory = _bracket_working_directory(working_directory_root, t_low, t_high)
        try:
            df = gather_calphy_results_detailed(working_directory)
        except FileNotFoundError:
            continue

        result_row = df.iloc[0]
        forward = result_row["forward_energy_diff"]
        backward = result_row["backward_energy_diff"]
        if forward is None or backward is None:
            continue

        criteria[(t_low, t_high)] = ts_overlap_criterion(forward[0], backward[0])

    return criteria


_BRACKET_LOG_FILENAME = "bracket_log.csv"


def _bracket_log_path(working_directory_root: str) -> str:
    return os.path.join(working_directory_root, _BRACKET_LOG_FILENAME)


def _read_bracket_log(working_directory_root: str) -> Optional[pd.DataFrame]:
    path = _bracket_log_path(working_directory_root)
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None


def _write_bracket_log(working_directory_root: str, log_df: pd.DataFrame) -> None:
    os.makedirs(working_directory_root, exist_ok=True)
    log_df.to_csv(_bracket_log_path(working_directory_root), index=False)


def load_bracket_history(
    working_directory_root: str,
) -> Tuple[List[Tuple[float, float]], Dict[Tuple[float, float], float]]:
    """Every bracket tried for this structure, and the criterion for each that has one.

    This is the log-backed replacement for scanning/re-parsing calphy output
    on every call: `bracket_log.csv` (a plain dataframe -- also handy for
    plotting criterion-vs-bracket later) is the source of truth once it
    exists, including brackets that have been submitted but not finished yet
    (recorded with a NaN criterion the moment they're submitted -- see
    `_record_bracket`), so a quick repeated call doesn't have to guess from
    directory timing whether a bracket is already in flight. Rather than
    leaning on an executor's own (occasionally unstable) submission cache
    for that, the log is this module's own explicit, on-disk bookkeeping.

    Falls back to rescanning calphy output directories directly (via
    `_find_tried_brackets`/`_tried_bracket_criteria`) when the log is
    missing or unreadable -- e.g. a `working_directory_root` populated
    before this log existed, or one where the log file was deleted
    independently of the calphy output it describes.
    """
    log_df = _read_bracket_log(working_directory_root)
    if log_df is not None:
        tried_brackets = list(zip(log_df["t_low"], log_df["t_high"]))
        criteria_by_bracket = {
            (row.t_low, row.t_high): row.criterion
            for row in log_df.itertuples()
            if pd.notna(row.criterion)
        }
        return tried_brackets, criteria_by_bracket

    tried_brackets = _find_tried_brackets(working_directory_root)
    criteria_by_bracket = _tried_bracket_criteria(working_directory_root, tried_brackets)
    return tried_brackets, criteria_by_bracket


def _record_bracket(
    working_directory_root: str,
    bracket: Tuple[float, float],
    criterion: Optional[float] = None,
) -> None:
    """Upsert one bracket's row in `bracket_log.csv` (NaN criterion = pending/unknown)."""
    log_df = _read_bracket_log(working_directory_root)
    if log_df is None:
        tried_brackets, criteria_by_bracket = load_bracket_history(working_directory_root)
        log_df = pd.DataFrame(
            [
                {"t_low": t_low, "t_high": t_high, "criterion": criteria_by_bracket.get((t_low, t_high))}
                for t_low, t_high in tried_brackets
            ],
            columns=["t_low", "t_high", "criterion"],
        )

    t_low, t_high = bracket
    mask = (log_df["t_low"] == t_low) & (log_df["t_high"] == t_high)
    if mask.any():
        if criterion is not None:
            log_df.loc[mask, "criterion"] = criterion
    else:
        log_df.loc[len(log_df), ["t_low", "t_high", "criterion"]] = [t_low, t_high, criterion]

    _write_bracket_log(working_directory_root, log_df)


def _calphy_parameters_for_bracket(
    calphy_parameters: Dict[str, Any],
    bracket: Tuple[float, float],
    initial_bracket: Tuple[float, float],
    scale_switching_steps_with_range: bool,
) -> Dict[str, Any]:
    """`calphy_parameters` with `n_switching_steps` scaled for `bracket`'s width.

    Defaults to on: without this, a bracket narrowed to a fraction of
    `initial_bracket`'s width would otherwise keep running the same
    switching-step count as the full-width bracket -- needlessly expensive
    once the bracket covers much less thermal ground. Scaled relative to
    `initial_bracket` specifically (not whatever bracket came right before),
    so the rate stays well-defined no matter how many narrowing steps have
    already happened. A no-op if `scale_switching_steps_with_range` is
    False or `calphy_parameters` has no `n_switching_steps`.
    """
    if not scale_switching_steps_with_range or "n_switching_steps" not in calphy_parameters:
        return calphy_parameters

    params = dict(calphy_parameters)
    params["n_switching_steps"] = scale_steps_to_bracket_width(
        calphy_parameters["n_switching_steps"], initial_bracket, bracket
    )
    return params


def submit_bracket(
    input_structure: Any,
    bracket: Tuple[float, float],
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor: Any,
    working_directory_root: str,
) -> Tuple[Any, str]:
    """Submit (or reconnect to) a specific, already-decided bracket.

    Pure executor plumbing: no tolerance, no convergence decision, no
    branching -- just "run this exact bracket." `tolerance` never appears
    here at all, so there is nothing in this function's inputs that a
    change in tolerance could possibly affect; it always submits/reconnects
    the same bracket for the same cache key regardless of what any caller's
    tolerance is doing.

    Parameters
    ----------
    input_structure : Any
        The structure to run TI on, e.g. an ase.Atoms.
    bracket : Tuple[float, float]
        The `(t_low, t_high)` bracket to submit.
    calphy_parameters : Dict[str, Any]
        Base calphy parameters; `temperature` is overwritten with `bracket`.
    potential_df : Any
        Potential DataFrame in pyiron-compatible format.
    executor : Any
        Object exposing `submit(fn, **kwargs) -> Future`, e.g. an
        executorlib executor.
    working_directory_root : str
        Directory (dedicated to this one structure) under which per-bracket
        working directories are created.

    Returns
    -------
    Tuple[Any, str]
        The submitted future, and the working directory it was submitted to.
    """
    working_directory = _bracket_working_directory(working_directory_root, bracket[0], bracket[1])

    params = dict(calphy_parameters)
    params["temperature"] = [bracket[0], bracket[1]]

    future = executor.submit(
        calc_free_energy_with_calphy,
        input_structure=input_structure,
        potential_df=potential_df,
        calphy_parameters=params,
        working_directory=working_directory,
    )

    return future, working_directory


def refine_temperature_bracket_manually(
    input_structure: Any,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor: Any,
    working_directory_root: str,
    initial_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    scale_switching_steps_with_range: bool = True,
) -> Dict[str, Any]:
    """Submit, check, and narrow a TI temperature bracket for one structure.

    The manual counterpart to `refine_temperature_bracket_with_chain`: instead of the whole
    narrowing sequence being submitted up front as a dependency chain, this
    expects to be called again by hand (e.g. from a notebook loop) each time
    you want to check in and, if needed, push the next narrower bracket out.

    Orchestrates two purpose-separated pieces: `submit_bracket` (executor
    plumbing, no tolerance) and `decide_next_bracket` (tolerance-based
    decision, no executor). Stateless across calls: recomputes the current
    bracket from `load_bracket_history`, this structure's on-disk bracket
    log (or, failing that, a rescan of calphy output directories).

    Deliberately never inspects the executor's `Future` at all -- not even
    `.done()`. executorlib's own background thread needs real wall-clock
    time to notice a result, even one that's already fully cached on disk,
    so checking `.done()` immediately after `.submit()` is always False
    regardless of whether the work is actually finished; it is not a
    meaningful signal. The only reliable, synchronous source of truth is
    disk state itself: if `gather_calphy_results_detailed` can already read
    a result for the current bracket, it's done; if not, this (re)submits it
    fire-and-forget (matching executorlib's documented disconnect pattern)
    and expects to be called again later, whenever that is.

    Parameters
    ----------
    input_structure : Any
        The structure to run TI on, e.g. an ase.Atoms.
    calphy_parameters : Dict[str, Any]
        Base calphy parameters; `temperature` is overwritten with the
        resolved bracket before submission.
    potential_df : Any
        Potential DataFrame in pyiron-compatible format.
    executor : Any
        Object exposing `submit(fn, **kwargs) -> Future`, e.g. an
        executorlib executor.
    working_directory_root : str
        Directory (dedicated to this one structure) under which per-bracket
        working directories are created and scanned.
    initial_bracket : Tuple[float, float]
        The `(t_low, t_high)` bracket to start from if nothing has been
        tried yet.
    tolerance : float
        Passed to `decide_next_bracket` to decide convergence. Never reaches
        `submit_bracket` or the executor.
    step_lower : Optional[float], optional
        If given, the lower bound is raised by this amount on non-convergence.
    step_upper : Optional[float], optional
        If given, the upper bound is lowered by this amount on non-convergence.
    scale_switching_steps_with_range : bool, optional
        If True (default), `calphy_parameters["n_switching_steps"]` is
        scaled for each submitted bracket so steps-per-degree stays constant
        as the bracket narrows -- see `_calphy_parameters_for_bracket`. Set
        False to run every bracket with the same step count regardless of
        width.

    Returns
    -------
    Dict[str, Any]
        `status` is one of:
        - "pending": no result on disk yet for the current bracket; it was
          just (re)submitted fire-and-forget (or was already pending from a
          prior call, in which case nothing is resubmitted). Call again
          later to check.
        - "incomplete": a result exists but forward/backward TI data isn't
          available (e.g. fe mode).
        - "converged": forward/backward overlap within tolerance.
        - "resubmitted": did not converge; a narrower bracket was submitted
          fire-and-forget. Call again later to check on it.
        Also includes `bracket`, `working_directory`, and (when available) `df`.
    """
    tried_brackets, criteria_by_bracket = load_bracket_history(working_directory_root)

    # Before doing anything with the narrowest tried bracket, check whether
    # some already-tried bracket -- possibly wider -- already satisfies
    # `tolerance`. Narrowing only ever moves in one direction once started;
    # if an earlier, wider bracket already converged, there's no reason to
    # wait on (or keep narrowing past) a narrower one, regardless of what
    # its own result says.
    best_converged = pick_best_converged_bracket(criteria_by_bracket, tolerance)
    if best_converged is not None:
        converged_working_directory = _bracket_working_directory(
            working_directory_root, best_converged[0], best_converged[1]
        )
        return {
            "status": "converged",
            "bracket": best_converged,
            "working_directory": converged_working_directory,
            "df": gather_calphy_results_detailed(converged_working_directory),
        }

    current_bracket = resolve_current_bracket(initial_bracket, tried_brackets)
    working_directory = _bracket_working_directory(working_directory_root, current_bracket[0], current_bracket[1])

    criterion = criteria_by_bracket.get(current_bracket)
    if criterion is None:
        try:
            df = gather_calphy_results_detailed(working_directory)
        except FileNotFoundError:
            if current_bracket not in tried_brackets:
                # Genuinely new: submit once and log it as pending right
                # away, so a quick repeated call -- before the job even
                # creates its working directory -- reports "pending"
                # instead of submitting the same calphy run a second time.
                # This is what replaces leaning on an executor's own
                # (occasionally unstable) submission cache for dedup.
                submit_bracket(
                    input_structure,
                    current_bracket,
                    _calphy_parameters_for_bracket(
                        calphy_parameters, current_bracket, initial_bracket, scale_switching_steps_with_range
                    ),
                    potential_df,
                    executor,
                    working_directory_root,
                )
                _record_bracket(working_directory_root, current_bracket)
            return {
                "status": "pending",
                "bracket": current_bracket,
                "working_directory": working_directory,
            }

        result_row = df.iloc[0]
        forward = result_row["forward_energy_diff"]
        backward = result_row["backward_energy_diff"]

        if forward is None or backward is None:
            return {
                "status": "incomplete",
                "bracket": current_bracket,
                "working_directory": working_directory,
                "df": df,
            }

        criterion = ts_overlap_criterion(forward[0], backward[0])
        _record_bracket(working_directory_root, current_bracket, criterion)
    else:
        df = gather_calphy_results_detailed(working_directory)

    next_bracket = decide_next_bracket(current_bracket, criterion, tolerance, step_lower=step_lower, step_upper=step_upper)

    if next_bracket is None:
        return {
            "status": "converged",
            "bracket": current_bracket,
            "working_directory": working_directory,
            "df": df,
        }

    new_working_directory = _bracket_working_directory(working_directory_root, next_bracket[0], next_bracket[1])
    submit_bracket(
        input_structure,
        next_bracket,
        _calphy_parameters_for_bracket(
            calphy_parameters, next_bracket, initial_bracket, scale_switching_steps_with_range
        ),
        potential_df,
        executor,
        working_directory_root,
    )
    _record_bracket(working_directory_root, next_bracket)

    return {
        "status": "resubmitted",
        "bracket": next_bracket,
        "working_directory": new_working_directory,
        "df": df,
    }


def _precompute_bracket_chain(
    current_bracket: Tuple[float, float],
    step_lower: Optional[float],
    step_upper: Optional[float],
    max_iterations: Optional[int],
) -> List[Tuple[float, float]]:
    """The full sequence of brackets a chain will attempt, decided up front.

    Always starts with `current_bracket` and applies `step_bracket` up to
    `max_iterations` more times. If a step would collapse the bracket (zero
    or negative width), the sequence just stops there instead of raising --
    deciding this eagerly, before any job is submitted, is what makes that a
    normal "the chain is shorter than max_iterations allows" outcome rather
    than an exception raised from inside an already-running, unobserved task
    (which is how a narrowing collapse could previously vanish silently).

    With `max_iterations=None` the chain runs until the bracket collapses;
    if that would take more than `_UNBOUNDED_CEILING` steps, raises instead
    of building (and then submitting) an absurdly long chain.
    """
    brackets = [current_bracket]
    bracket = current_bracket
    for _ in range(_UNBOUNDED_CEILING if max_iterations is None else max_iterations):
        try:
            bracket = step_bracket(bracket[0], bracket[1], step_lower=step_lower, step_upper=step_upper)
        except ValueError:
            break
        brackets.append(bracket)
    else:
        if max_iterations is None:
            raise ValueError(
                f"max_iterations=None would submit more than {_UNBOUNDED_CEILING} brackets with this step size; "
                "set a finite max_iterations."
            )
    return brackets


def _run_bracket_chain_step(
    input_structure: Any,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    working_directory_root: str,
    bracket: Tuple[float, float],
    initial_bracket: Tuple[float, float],
    tolerance: float,
    scale_switching_steps_with_range: bool,
    previous_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one bracket in a dependency chain, or pass through if there's nothing to do.

    This is what actually gets submitted for every bracket in the chain
    built by `refine_temperature_bracket_with_chain` -- never `calc_free_energy_with_calphy`
    directly. `previous_result` is the *resolved value* of the previous
    step's Future: executorlib's dependency resolution substitutes it in
    automatically once that Future completes, so this function never has to
    poll or wait for it itself.

    If the previous step already reached a terminal state (`"converged"`,
    `"incomplete"`, or ran out of chain), this step is a no-op: it just
    returns that same result unchanged, so every later step in a
    precomputed chain collapses to a cheap pass-through once the real work
    is done. Otherwise it behaves exactly like the corresponding step in
    `refine_temperature_bracket_manually`: reuse an already-computed result
    for this bracket if one exists on disk (bootstrapping a second,
    independent chain must never silently rerun calphy's stochastic MD and
    risk a different criterion the second time), else run it for real and
    record the result.
    """
    if previous_result is not None and previous_result["status"] != "narrowed":
        return previous_result

    working_directory = _bracket_working_directory(working_directory_root, bracket[0], bracket[1])
    _, criteria_by_bracket = load_bracket_history(working_directory_root)

    # Approximate equality: t_low/t_high round-trip through the .2f
    # directory-naming format, so a bracket combined or stepped in-memory
    # can differ from the parsed disk value in the last decimal without
    # being a genuinely different bracket.
    already_computed = any(
        np.isclose(tried[0], bracket[0], atol=0.005) and np.isclose(tried[1], bracket[1], atol=0.005)
        for tried in criteria_by_bracket
    )
    if already_computed:
        df = gather_calphy_results_detailed(working_directory)
    else:
        params = dict(calphy_parameters)
        params["temperature"] = list(bracket)
        params = _calphy_parameters_for_bracket(params, bracket, initial_bracket, scale_switching_steps_with_range)
        _, df = calc_free_energy_with_calphy(
            input_structure=input_structure,
            potential_df=potential_df,
            calphy_parameters=params,
            working_directory=working_directory,
        )

    result_row = df.iloc[0]
    forward = result_row["forward_energy_diff"]
    backward = result_row["backward_energy_diff"]

    if forward is None or backward is None:
        return {"status": "incomplete", "bracket": bracket, "working_directory": working_directory, "df": df}

    criterion = ts_overlap_criterion(forward[0], backward[0])
    _record_bracket(working_directory_root, bracket, criterion)

    status = "converged" if criterion <= tolerance else "narrowed"
    return {"status": status, "bracket": bracket, "working_directory": working_directory, "df": df}


def refine_temperature_bracket_with_chain(
    input_structure: Any,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor: Any,
    working_directory_root: str,
    initial_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    max_iterations: Optional[int] = DEFAULT_MAX_ITERATIONS,
    scale_switching_steps_with_range: bool = True,
) -> Tuple[List[Any], str]:
    """Submit a whole bracket-narrowing sequence up front, chained via executor dependencies.

    Unlike calling `refine_temperature_bracket_manually` by hand repeatedly,
    this precomputes the full candidate bracket sequence (see
    `_precompute_bracket_chain`) and submits every step to `executor` at
    once: the first with no dependency, and each later step with the
    previous step's own Future passed straight through as its
    `previous_result` argument. executorlib resolves that dependency for
    you -- `SingleNodeExecutor` and `SlurmClusterExecutor` both wrap a
    `DependencyTaskScheduler` by default, so passing a Future as a submit
    argument makes executorlib wait for it and substitute its resolved
    value before running the dependent task. For a cluster executor, this
    dependency is even held by the scheduler itself (e.g. SLURM's own
    `--dependency=afterok:<jobid>`, via pysqa), so nothing of ours has to
    stay running for the chain to complete -- genuinely submit and walk
    away. For `SingleNodeExecutor`, the same as before applies: the calling
    process has to stay alive for its dependency-resolution thread to keep
    driving the chain.

    Each step is cheap once the real work upstream is done: a step whose
    predecessor already converged, hit "incomplete", or was the chain's
    last precomputed bracket just returns that same result unchanged (see
    `_run_bracket_chain_step`) rather than doing anything further.

    Parameters
    ----------
    input_structure : Any
        The structure to run TI on, e.g. an ase.Atoms.
    calphy_parameters, potential_df, working_directory_root, initial_bracket,
    tolerance, step_lower, step_upper, scale_switching_steps_with_range :
        Same meaning as in `refine_temperature_bracket_manually`.
    executor : Any
        Object exposing `submit(fn, **kwargs) -> Future` with executorlib's
        dependency-resolution behavior (the default for `SingleNodeExecutor`,
        `SlurmClusterExecutor`, and `FluxClusterExecutor` alike) -- every
        step in the chain is submitted to this one executor.
    max_iterations : Optional[int], optional
        Maximum number of narrowing steps beyond the first bracket (default
        5). Bounds the precomputed chain to at most `max_iterations + 1`
        brackets; a narrowing step that would collapse the bracket ends the
        chain earlier than that, without raising. `None` means "until the
        bracket collapses" and warns: every bracket is a full job.

    Returns
    -------
    Tuple[List[Any], str]
        The list of Futures, one per bracket in the precomputed chain (in
        order), and the first bracket's working directory. Call `.result()`
        on the *last* Future to wait for the whole chain to settle. If some
        already-tried bracket -- possibly wider than whatever's narrowest on
        disk -- already satisfies `tolerance` (see `pick_best_converged_bracket`),
        nothing is submitted at all: a single already-resolved Future for
        that bracket is returned instead, so re-running with a looser
        tolerance than a previous session recovers the cheapest bracket that
        was already good enough, rather than resuming from the narrowest one.
    """
    _check_max_iterations(max_iterations)

    tried_brackets, criteria_by_bracket = load_bracket_history(working_directory_root)

    # Same check refine_temperature_bracket_manually makes: narrowing only
    # ever moves in one direction once started, so if an earlier, wider
    # bracket already satisfies today's tolerance, there's no reason to
    # start (or continue) a chain from whatever's narrowest on disk --
    # that's how a looser tolerance, applied later, recovers a cheaper
    # bracket instead of always re-deriving the narrowest one ever reached.
    best_converged = pick_best_converged_bracket(criteria_by_bracket, tolerance)
    if best_converged is not None:
        converged_working_directory = _bracket_working_directory(
            working_directory_root, best_converged[0], best_converged[1]
        )
        converged_future: Any = Future()
        converged_future.set_result(
            {
                "status": "converged",
                "bracket": best_converged,
                "working_directory": converged_working_directory,
                "df": gather_calphy_results_detailed(converged_working_directory),
            }
        )
        return [converged_future], converged_working_directory

    current_bracket = resolve_current_bracket(initial_bracket, tried_brackets)
    bracket_chain = _precompute_bracket_chain(current_bracket, step_lower, step_upper, max_iterations)

    futures: List[Any] = []
    previous_result = None
    for bracket in bracket_chain:
        future = executor.submit(
            _run_bracket_chain_step,
            input_structure=input_structure,
            calphy_parameters=calphy_parameters,
            potential_df=potential_df,
            working_directory_root=working_directory_root,
            bracket=bracket,
            initial_bracket=initial_bracket,
            tolerance=tolerance,
            scale_switching_steps_with_range=scale_switching_steps_with_range,
            previous_result=previous_result,
        )
        futures.append(future)
        previous_result = future

    working_directory = _bracket_working_directory(working_directory_root, bracket_chain[0][0], bracket_chain[0][1])
    return futures, working_directory


DEFAULT_STALE_AFTER_SECONDS = 6 * 3600

_CONFIG_FILENAME = "ts_config.json"

def _config_fingerprint(
    input_structure: Any,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    initial_bracket: Tuple[float, float],
    scale_switching_steps_with_range: bool,
) -> str:
    """Hash everything that changes what a bracket's result means.

    Deliberately excludes the temperature bracket itself (varies per bracket), the tolerance
    and the step sizes (deciding what to do with a result, not producing it).
    """
    hasher = hashlib.sha256()
    hasher.update(np.asarray(input_structure.get_atomic_numbers()).tobytes())
    hasher.update(np.round(input_structure.get_positions(), 8).tobytes())
    hasher.update(np.round(np.asarray(input_structure.get_cell()), 8).tobytes())
    hasher.update(np.asarray(input_structure.get_pbc()).tobytes())
    parameters = {key: value for key, value in calphy_parameters.items() if key not in ("temperature", "lattice")}
    hasher.update(
        json.dumps(
            {
                "parameters": parameters,
                "potential": potential_df[["Config", "Species"]].to_dict("list"),
                "initial_bracket": list(initial_bracket),
                "scale_switching_steps_with_range": scale_switching_steps_with_range,
            },
            sort_keys=True,
            default=str,
        ).encode()
    )
    return hasher.hexdigest()


def _check_config(working_directory_root: str, fingerprint: str) -> None:
    """Pin this folder to one configuration; refuse to mix results from different ones.

    The first call writes ``ts_config.json``; every later call must present the same fingerprint.
    Without this, changing e.g. ``n_switching_steps`` and re-running in the same folder would
    silently reuse results computed with the old value.
    """
    path = os.path.join(working_directory_root, _CONFIG_FILENAME)
    if os.path.isfile(path):
        with open(path) as handle:
            stored = json.load(handle)["fingerprint"]
        if stored != fingerprint:
            raise ValueError(
                f"{working_directory_root} was created with a different structure, potential or calphy "
                f"parameters (see {_CONFIG_FILENAME}). Use a new working_directory_root, or delete that "
                f"folder's results if you really mean to start over."
            )
        return
    os.makedirs(working_directory_root, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"fingerprint": fingerprint}, handle)


def _marker_path(working_directory_root: str, kind: str, bracket: Tuple[float, float]) -> str:
    return os.path.join(working_directory_root, f".{kind}_T_{bracket[0]:.2f}_{bracket[1]:.2f}")


def _bracket_cache_key(working_directory_root: str, bracket: Tuple[float, float]) -> str:
    """Executor task name for the job that works on `bracket` of the structure in `working_directory_root`.

    executorlib names a task by a hash of the function and its arguments, and the bracket is not
    an argument (the job finds it in the bracket log). Every job after the first would therefore get
    the same name as its predecessor, and the SLURM executor does not submit a task whose name belongs
    to a job that is still running -- the hand-over would silently vanish.

    The structure folder stands for structure, potential and calphy parameters (`ts_config.json`
    refuses a folder used with different ones), so folder + bracket identify the unit of work.
    """
    root_hash = hashlib.sha1(os.path.abspath(working_directory_root).encode()).hexdigest()[:12]
    return f"refine_temperature_bracket_T_{bracket[0]:.2f}_{bracket[1]:.2f}_{root_hash}"


def _bracket_resource_dict(
    working_directory_root: str, bracket: Tuple[float, float], executor_spec: ExecutorSpec
) -> Dict[str, str]:
    """`resource_dict` naming the task for `bracket`, after removing a result left from an earlier attempt.

    The bracket log, not executorlib's cache, says what has run. A leftover `<name>_o.h5` (the job
    raised, or a bracket that stopped at `max_iterations` is being continued) would make executorlib
    treat the task as done and submit nothing. A job still running has no result file yet, so it is
    unaffected: executorlib still declines to submit the same bracket twice at the same time.
    """
    cache_key = _bracket_cache_key(working_directory_root, bracket)
    cache_directory = os.path.abspath(executor_spec.kwargs.get("cache_directory", "executorlib_cache"))
    stale_result = os.path.join(cache_directory, cache_key + "_o.h5")
    if os.path.exists(stale_result):
        os.remove(stale_result)
    return {"cache_key": cache_key}


def current_bracket_resource_dict(
    working_directory_root: str, initial_bracket: Tuple[float, float], executor_spec: ExecutorSpec
) -> Dict[str, str]:
    """`resource_dict` for submitting `refine_temperature_bracket_with_resubmission` for one structure.

    Pass it as ``executor.submit(..., resource_dict=...)`` from a notebook. It names the task after
    the structure folder and the bracket the call will work on (the last one in the bracket log, or
    `initial_bracket` for a new folder), so re-running the submit cell to resume a structure starts a
    job, and an accidental second run while that job is still running does not start another.
    Side effect: a result the executor cached for that name by an earlier attempt is removed.
    """
    tried_brackets, _ = load_bracket_history(working_directory_root)
    bracket = resolve_current_bracket(initial_bracket, tried_brackets)
    return _bracket_resource_dict(working_directory_root, bracket, executor_spec)


def _claim(path: str) -> bool:
    """Create `path` exclusively. True if this call created it, False if it already existed."""
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"time": time.time(), "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, handle)
    return True


def _last_activity(working_directory: str, marker: str) -> float:
    """Most recent modification time of the start marker or anything in the bracket's folder."""
    times = [os.path.getmtime(marker)] if os.path.exists(marker) else []
    if os.path.isdir(working_directory):
        times += [entry.stat().st_mtime for entry in os.scandir(working_directory)]
    return max(times) if times else 0.0


def _read_criterion(working_directory: str) -> Optional[float]:
    """The bracket's TI overlap criterion, or None if it has no complete result yet."""
    try:
        row = gather_calphy_results_detailed(working_directory).iloc[0]
        if not row["status"] or row["forward_energy_diff"] is None or row["backward_energy_diff"] is None:
            return None
        return ts_overlap_criterion(row["forward_energy_diff"][0], row["backward_energy_diff"][0])
    except (FileNotFoundError, OSError, ValueError, KeyError, yaml.YAMLError):
        # a missing or half-written result is "not finished", not an error
        return None


def _outcome(status: str, bracket: Tuple[float, float], working_directory_root: str, **extra: Any) -> Dict[str, Any]:
    return {
        "status": status,
        "bracket": (float(bracket[0]), float(bracket[1])),
        "working_directory": _bracket_working_directory(working_directory_root, bracket[0], bracket[1]),
        **extra,
    }


def refine_temperature_bracket_with_resubmission(
    input_structure: Any,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor_spec: ExecutorSpec,
    working_directory_root: str,
    initial_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    scale_switching_steps_with_range: bool = True,
    max_iterations: Optional[int] = DEFAULT_MAX_ITERATIONS,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    force: bool = False,
) -> Dict[str, Any]:
    """Run one bracket of a structure's TS convergence here, then hand the rest on.

    Submit this function itself (through ``executor_spec.create()``) to start a structure; it
    runs the next bracket that still has no result, records its criterion in
    ``bracket_log.csv``, and -- if the bracket did not converge -- submits itself again for the
    next, narrower bracket and returns. Every step is its own job with its own wall time, and
    nothing has to stay running in between. The first submission can go through any executor
    (even a local one that finishes in seconds, since submitting is all it does); the notebook can
    be closed afterwards.

    The state lives entirely on disk, so calling this again with the same arguments and the same
    ``working_directory_root`` resumes: brackets that already finished are skipped, and the first
    one without a result is run. Only do that once nothing of this structure is queued or running
    any more; a bracket whose folder saw file activity within ``stale_after_seconds`` is left
    alone (``"running"``) unless ``force=True``, but a job that is merely still waiting in the
    queue leaves no trace on disk and cannot be told apart from a dead one.

    Parameters
    ----------
    input_structure : Any
        The structure to run TI on, e.g. an ase.Atoms.
    calphy_parameters : Dict[str, Any]
        Base calphy parameters; ``temperature`` is overwritten per bracket.
    potential_df : Any
        Potential DataFrame in pyiron-compatible format.
    executor_spec : ExecutorSpec
        Recipe for the executor the next bracket is submitted to (this job builds its own).
    working_directory_root : str
        Folder dedicated to this one structure; per-bracket folders, ``bracket_log.csv`` and
        ``ts_config.json`` live here. Pinned to one configuration (see ``ValueError`` below).
    initial_bracket : Tuple[float, float]
        The ``(t_low, t_high)`` bracket to start from.
    tolerance : float
        Maximum TI overlap criterion for a bracket to count as converged.
    step_lower, step_upper : Optional[float], optional
        How far to raise the lower / lower the upper bound on non-convergence. At least one is
        required.
    scale_switching_steps_with_range : bool, optional
        As in ``refine_temperature_bracket_manually``: keep steps per kelvin constant as the
        bracket narrows (default) or run every bracket with the same step count.
    max_iterations : Optional[int], optional
        Cap on narrowing steps beyond the first bracket, so at most ``max_iterations + 1`` brackets are run
        for this folder (default 5). Counted over the whole folder, so a re-run to go further needs a larger
        value. ``None`` means "until converged or the bracket can no longer be narrowed" and warns, since
        every bracket is a full job (safety ceiling: 1000 brackets).
    stale_after_seconds : float, optional
        How long a bracket's folder must have been silent before a re-run takes it over.
    force : bool, optional
        Take over a bracket even if its folder shows recent activity.

    Returns
    -------
    Dict[str, Any]
        Small dict (it is stored by the executor as the job's result) with ``status``, ``bracket``
        and ``working_directory``. ``status`` is one of:

        - ``"converged"``: a bracket met the tolerance; nothing more to do.
        - ``"exhausted"``: the next narrowing step would collapse the bracket -- no temperature
          range left that meets the tolerance.
        - ``"limit_reached"``: this bracket did not converge and ``max_iterations`` is used up; nothing more
          was submitted.
        - ``"resubmitted"``: this bracket did not converge; the next one was submitted.
        - ``"running"``: this bracket's folder shows recent activity, so it was left alone.

    Raises
    ------
    ValueError
        If neither step is given, ``max_iterations`` is neither a non-negative integer nor None, or ``working_directory_root`` was created with a different
        structure, potential or calphy parameters.
    """
    if step_lower is None and step_upper is None:
        raise ValueError("At least one of step_lower or step_upper must be given.")
    _check_max_iterations(max_iterations)

    _check_config(
        working_directory_root,
        _config_fingerprint(input_structure, calphy_parameters, potential_df, initial_bracket, scale_switching_steps_with_range),
    )

    tried_brackets, criteria_by_bracket = load_bracket_history(working_directory_root)

    best_converged = pick_best_converged_bracket(criteria_by_bracket, tolerance)
    if best_converged is not None:
        return _outcome("converged", best_converged, working_directory_root)

    bracket = resolve_current_bracket(initial_bracket, tried_brackets)
    working_directory = _bracket_working_directory(working_directory_root, bracket[0], bracket[1])

    criterion = criteria_by_bracket.get(bracket)
    if criterion is None:
        # Finished on disk but never logged (e.g. the job died right after calphy did)?
        criterion = _read_criterion(working_directory)
        if criterion is not None:
            _record_bracket(working_directory_root, bracket, criterion)

    if criterion is None:
        start_marker = _marker_path(working_directory_root, "started", bracket)
        if not _claim(start_marker):
            idle_seconds = time.time() - _last_activity(working_directory, start_marker)
            if not force and idle_seconds <= stale_after_seconds:
                return _outcome("running", bracket, working_directory_root, idle_seconds=idle_seconds)
            os.remove(start_marker)
            _claim(start_marker)

        _record_bracket(working_directory_root, bracket)
        parameters = dict(calphy_parameters)
        parameters["temperature"] = [float(bracket[0]), float(bracket[1])]
        parameters = _calphy_parameters_for_bracket(parameters, bracket, initial_bracket, scale_switching_steps_with_range)
        try:
            calc_free_energy_with_calphy(
                input_structure=input_structure,
                potential_df=potential_df,
                calphy_parameters=parameters,
                working_directory=working_directory,
            )
        except BaseException:
            # a job that raised is known to be dead: let a re-run start right away instead of
            # waiting out `stale_after_seconds` (a job killed by the scheduler cannot do this)
            os.remove(start_marker)
            raise

        criterion = _read_criterion(working_directory)
        if criterion is None:
            raise RuntimeError(
                f"calphy finished in {working_directory} but left no forward/backward energy differences "
                "(is calphy_parameters['mode'] 'ts'?)"
            )
        _record_bracket(working_directory_root, bracket, criterion)

    try:
        next_bracket = decide_next_bracket(bracket, criterion, tolerance, step_lower=step_lower, step_upper=step_upper)
    except ValueError:
        return _outcome("exhausted", bracket, working_directory_root, criterion=float(criterion))

    if next_bracket is None:
        return _outcome("converged", bracket, working_directory_root, criterion=float(criterion))

    if len(load_bracket_history(working_directory_root)[0]) > (_UNBOUNDED_CEILING if max_iterations is None else max_iterations):
        return _outcome("limit_reached", bracket, working_directory_root, criterion=float(criterion))

    _submit_next_bracket(
        next_bracket,
        executor_spec,
        working_directory_root,
        dict(
            input_structure=input_structure,
            calphy_parameters=calphy_parameters,
            potential_df=potential_df,
            executor_spec=executor_spec,
            working_directory_root=working_directory_root,
            initial_bracket=initial_bracket,
            tolerance=tolerance,
            step_lower=step_lower,
            step_upper=step_upper,
            scale_switching_steps_with_range=scale_switching_steps_with_range,
            max_iterations=max_iterations,
            stale_after_seconds=stale_after_seconds,
        ),
    )
    return _outcome("resubmitted", next_bracket, working_directory_root, criterion=float(criterion))


def _submit_next_bracket(
    next_bracket: Tuple[float, float],
    executor_spec: ExecutorSpec,
    working_directory_root: str,
    kwargs: Dict[str, Any],
) -> None:
    """Submit `refine_temperature_bracket_with_resubmission` for the next bracket, at most once per bracket.

    The exclusive claim file is what prevents a duplicate: if this bracket was already handed on
    (this job re-run by the scheduler, or a second run of the same structure), nothing is
    submitted again -- independent of any executor-side cache.
    """
    claim = _marker_path(working_directory_root, "submitted", next_bracket)
    if not _claim(claim):
        return

    _record_bracket(working_directory_root, next_bracket)
    try:
        executor = executor_spec.create()
        future = executor.submit(
            refine_temperature_bracket_with_resubmission,
            resource_dict=_bracket_resource_dict(working_directory_root, next_bracket, executor_spec),
            **kwargs,
        )
        if executor_spec.detaches:
            executor.shutdown(wait=False)
        else:
            future.result()
            executor.shutdown(wait=True)
    except BaseException:
        os.remove(claim)
        raise
