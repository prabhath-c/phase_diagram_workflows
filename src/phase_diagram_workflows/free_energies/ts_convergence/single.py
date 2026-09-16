from __future__ import annotations

import os
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

    The manual counterpart to `submit_bracket_chain`: instead of the whole
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
    max_iterations: int,
) -> List[Tuple[float, float]]:
    """The full sequence of brackets a chain will attempt, decided up front.

    Always starts with `current_bracket` and applies `step_bracket` up to
    `max_iterations` more times. If a step would collapse the bracket (zero
    or negative width), the sequence just stops there instead of raising --
    deciding this eagerly, before any job is submitted, is what makes that a
    normal "the chain is shorter than max_iterations allows" outcome rather
    than an exception raised from inside an already-running, unobserved task
    (which is how a narrowing collapse could previously vanish silently).
    """
    brackets = [current_bracket]
    bracket = current_bracket
    for _ in range(max_iterations):
        try:
            bracket = step_bracket(bracket[0], bracket[1], step_lower=step_lower, step_upper=step_upper)
        except ValueError:
            break
        brackets.append(bracket)
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
    built by `submit_bracket_chain` -- never `calc_free_energy_with_calphy`
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


def submit_bracket_chain(
    input_structure: Any,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor: Any,
    working_directory_root: str,
    initial_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    max_iterations: int = 10,
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
    max_iterations : int, optional
        Maximum number of narrowing steps beyond the first bracket. Bounds
        the precomputed chain to at most `max_iterations + 1` brackets; a
        narrowing step that would collapse the bracket ends the chain
        earlier than that, without raising.

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
