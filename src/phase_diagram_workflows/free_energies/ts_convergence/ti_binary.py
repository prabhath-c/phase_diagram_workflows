from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from phase_diagram_workflows.free_energies.ti_calculator import (
    calc_free_energy_with_calphy,
    gather_calphy_results_detailed,
)
from phase_diagram_workflows.free_energies.ts_convergence.base import (
    decide_next_bracket,
    pick_best_converged_bracket,
    resolve_current_bracket,
    ts_overlap_criterion,
)

_IDENTITY_COLUMNS = ["main_element", "mixing_element", "phase_type", "reference_phase", "c_in"]


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting functions in ts_convergence.ti_binary. "
            "Install it, e.g. `pip install matplotlib`."
        )


def _bracket_prefix(row: pd.Series, conc_decimals: int = 6) -> str:
    """Build the naming prefix identifying a (concentration, phase) row.

    Mirrors jackall's job-naming convention, applied to a working-directory
    prefix instead of a pyiron job name.
    """
    elements = f"{row['main_element']}{row['mixing_element']}"
    conc = f"{row['c_in']:.{conc_decimals}f}"
    return f"{elements}_{row['phase_type']}_{row['reference_phase']}_{conc}"


def _bracket_working_directory(working_directory_root: str, prefix: str, t_low: float, t_high: float) -> str:
    return os.path.join(working_directory_root, f"{prefix}_T_{t_low:.2f}_{t_high:.2f}")


def _find_tried_brackets(working_directory_root: str, prefix: str) -> List[Tuple[float, float]]:
    """Recover every bracket already attempted for `prefix` by scanning disk.

    This is the stateless replacement for pyiron's job table: the current
    bracket for a row is always recomputed from what's already on disk,
    never stored separately.
    """
    marker = f"{prefix}_T_"
    tried: List[Tuple[float, float]] = []
    if not os.path.isdir(working_directory_root):
        return tried

    for name in os.listdir(working_directory_root):
        if not name.startswith(marker):
            continue
        if not os.path.isdir(os.path.join(working_directory_root, name)):
            continue

        parts = name[len(marker):].split("_")
        if len(parts) != 2:
            continue  # unparsable suffix (e.g. a manual backup folder) -- skip, don't crash
        try:
            t_low, t_high = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        tried.append((t_low, t_high))

    return tried


def _tried_bracket_criteria(
    prefix: str,
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
        working_directory = _bracket_working_directory(working_directory_root, prefix, t_low, t_high)
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


def submit_bracket(
    row: pd.Series,
    bracket: Tuple[float, float],
    calphy_parameters: Dict[str, Any],
    potential_df: pd.DataFrame,
    executor: Any,
    working_directory_root: str,
    conc_decimals: int = 6,
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
    row : pd.Series
        One row of a structures DataFrame, providing `main_element`,
        `mixing_element`, `phase_type`, `reference_phase`, `c_in`, `atoms`.
    bracket : Tuple[float, float]
        The `(t_low, t_high)` bracket to submit.
    calphy_parameters : Dict[str, Any]
        Base calphy parameters; `temperature` is overwritten with `bracket`.
    potential_df : pd.DataFrame
        Potential DataFrame in pyiron-compatible format.
    executor : Any
        Object exposing `submit(fn, **kwargs) -> Future`, e.g. an
        executorlib executor.
    working_directory_root : str
        Directory under which per-bracket working directories are created.
    conc_decimals : int, optional
        Decimal precision used when formatting `c_in` into the naming prefix.

    Returns
    -------
    Tuple[Any, str]
        The submitted future, and the working directory it was submitted to.
    """
    prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
    working_directory = _bracket_working_directory(working_directory_root, prefix, bracket[0], bracket[1])

    params = dict(calphy_parameters)
    params["temperature"] = [bracket[0], bracket[1]]

    future = executor.submit(
        calc_free_energy_with_calphy,
        input_structure=row["atoms"],
        potential_df=potential_df,
        calphy_parameters=params,
        working_directory=working_directory,
    )

    return future, working_directory


def refine_temperature_bracket(
    row: pd.Series,
    calphy_parameters: Dict[str, Any],
    potential_df: pd.DataFrame,
    executor: Any,
    working_directory_root: str,
    initial_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    conc_decimals: int = 6,
) -> Dict[str, Any]:
    """Submit, check, and narrow a TI temperature bracket for one structure row.

    Orchestrates two purpose-separated pieces: `submit_bracket` (executor
    plumbing, no tolerance) and `decide_next_bracket` (tolerance-based
    decision, no executor). Stateless: recomputes the current bracket from
    whatever brackets have already been attempted on disk (see
    `_find_tried_brackets`).

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
    row : pd.Series
        One row of a structures DataFrame (e.g. from
        `generate_random_binary_structures`), providing `main_element`,
        `mixing_element`, `phase_type`, `reference_phase`, `c_in`, `atoms`.
    calphy_parameters : Dict[str, Any]
        Base calphy parameters; `temperature` is overwritten with the
        resolved bracket before submission.
    potential_df : pd.DataFrame
        Potential DataFrame in pyiron-compatible format.
    executor : Any
        Object exposing `submit(fn, **kwargs) -> Future`, e.g. an
        executorlib executor.
    working_directory_root : str
        Directory under which per-bracket working directories are created
        and scanned.
    initial_bracket : Tuple[float, float]
        The `(t_low, t_high)` bracket to start from if nothing has been
        tried yet for this row.
    tolerance : float
        Passed to `decide_next_bracket` to decide convergence. Never reaches
        `submit_bracket` or the executor.
    step_lower : Optional[float], optional
        If given, the lower bound is raised by this amount on non-convergence.
    step_upper : Optional[float], optional
        If given, the upper bound is lowered by this amount on non-convergence.
    conc_decimals : int, optional
        Decimal precision used when formatting `c_in` into the naming prefix.

    Returns
    -------
    Dict[str, Any]
        `status` is one of:
        - "pending": no result on disk yet for the current bracket; it was
          just (re)submitted fire-and-forget. Call again later to check.
        - "incomplete": a result exists but forward/backward TI data isn't
          available (e.g. fe mode).
        - "converged": forward/backward overlap within tolerance.
        - "resubmitted": did not converge; a narrower bracket was submitted
          fire-and-forget. Call again later to check on it.
        Also includes `bracket`, `working_directory`, and (when available) `df`.
    """
    prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
    tried_brackets = _find_tried_brackets(working_directory_root, prefix)

    # Before doing anything with the narrowest tried bracket, check whether
    # some already-tried bracket -- possibly wider -- already satisfies
    # `tolerance`. Narrowing only ever moves in one direction once started;
    # if an earlier, wider bracket already converged, there's no reason to
    # wait on (or keep narrowing past) a narrower one, regardless of what
    # its own result says.
    criteria_by_bracket = _tried_bracket_criteria(prefix, working_directory_root, tried_brackets)
    best_converged = pick_best_converged_bracket(criteria_by_bracket, tolerance)
    if best_converged is not None:
        converged_working_directory = _bracket_working_directory(
            working_directory_root, prefix, best_converged[0], best_converged[1]
        )
        return {
            "status": "converged",
            "bracket": best_converged,
            "working_directory": converged_working_directory,
            "df": gather_calphy_results_detailed(converged_working_directory),
        }

    current_bracket = resolve_current_bracket(initial_bracket, tried_brackets)
    working_directory = _bracket_working_directory(working_directory_root, prefix, current_bracket[0], current_bracket[1])

    try:
        df = gather_calphy_results_detailed(working_directory)
    except FileNotFoundError:
        submit_bracket(
            row, current_bracket, calphy_parameters, potential_df, executor, working_directory_root, conc_decimals
        )
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
    next_bracket = decide_next_bracket(current_bracket, criterion, tolerance, step_lower=step_lower, step_upper=step_upper)

    if next_bracket is None:
        return {
            "status": "converged",
            "bracket": current_bracket,
            "working_directory": working_directory,
            "df": df,
        }

    new_working_directory = _bracket_working_directory(working_directory_root, prefix, next_bracket[0], next_bracket[1])
    submit_bracket(
        row, next_bracket, calphy_parameters, potential_df, executor, working_directory_root, conc_decimals
    )

    return {
        "status": "resubmitted",
        "bracket": next_bracket,
        "working_directory": new_working_directory,
        "df": df,
    }


def auto_refine_temperature_brackets(
    structures_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any],
    potential_df: pd.DataFrame,
    executor: Any,
    working_directory_root: str,
    initial_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    conc_decimals: int = 6,
    max_iterations: int = 10,
) -> pd.DataFrame:
    """Drive every row of a structures DataFrame to convergence automatically.

    Unlike `refine_temperature_bracket`, which never touches the executor's
    `Future` and expects to be called again by hand whenever you want to
    check in, this blocks for real completion using the pattern executorlib
    is actually designed around: submit, then call `.result()`, never poll
    `.done()` immediately after submitting.

    Rows progress fully independently of each other: every row is submitted
    once up front, and then, using `wait(..., return_when=FIRST_COMPLETED)`,
    whichever row's job finishes first is immediately decided and (if not
    converged) resubmitted on its own -- every other row's job keeps running
    untouched in the background. A row never waits on a sibling it has no
    working-directory/cache relationship with; `max_iterations` is a per-row
    cap on narrowing attempts, not a global round count.

    `tolerance` here doesn't have to be right: nothing on disk is ever
    destroyed, and every attempted bracket stays inspectable via
    `build_criterion_table` regardless of what this function decided. A row
    that never converges within `max_iterations` just keeps its last status
    (still "resubmitted", with its narrowest bracket left running) -- this
    does not raise, it simply stops resubmitting that row.

    Parameters
    ----------
    structures_df : pd.DataFrame
        Structures DataFrame, one row per concentration/phase.
    calphy_parameters, potential_df, executor, working_directory_root,
    initial_bracket, tolerance, step_lower, step_upper, conc_decimals :
        Same meaning as in `refine_temperature_bracket`.
    max_iterations : int, optional
        Maximum number of narrowing attempts *per row* before giving up on
        that row specifically. Guards against a tolerance that's unreachable
        for that particular concentration (endless narrowing).

    Returns
    -------
    pd.DataFrame
        One row per input row (same index), with columns `status`
        (`"converged"`, `"incomplete"`, or `"resubmitted"` if
        `max_iterations` was reached first), `bracket`, `working_directory`,
        and `df` where available.
    """
    results: Dict[Any, Dict[str, Any]] = {}
    current_bracket_by_idx: Dict[Any, Tuple[float, float]] = {}
    iterations_by_idx: Dict[Any, int] = {idx: 0 for idx in structures_df.index}
    future_to_idx: Dict[Any, Any] = {}

    for idx in structures_df.index:
        row = structures_df.loc[idx]
        prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
        tried_brackets = _find_tried_brackets(working_directory_root, prefix)
        current_bracket_by_idx[idx] = resolve_current_bracket(initial_bracket, tried_brackets)
        future, _ = submit_bracket(
            row, current_bracket_by_idx[idx], calphy_parameters, potential_df,
            executor, working_directory_root, conc_decimals,
        )
        future_to_idx[future] = idx

    while future_to_idx:
        done, _ = wait(list(future_to_idx.keys()), return_when=FIRST_COMPLETED)

        for future in done:
            idx = future_to_idx.pop(future)
            row = structures_df.loc[idx]
            prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
            current_bracket = current_bracket_by_idx[idx]
            working_directory = _bracket_working_directory(
                working_directory_root, prefix, current_bracket[0], current_bracket[1]
            )

            _, df = future.result()
            result_row = df.iloc[0]
            forward = result_row["forward_energy_diff"]
            backward = result_row["backward_energy_diff"]

            if forward is None or backward is None:
                results[idx] = {
                    "status": "incomplete",
                    "bracket": current_bracket,
                    "working_directory": working_directory,
                    "df": df,
                }
                continue

            criterion = ts_overlap_criterion(forward[0], backward[0])
            next_bracket = decide_next_bracket(
                current_bracket, criterion, tolerance, step_lower=step_lower, step_upper=step_upper
            )

            if next_bracket is None:
                results[idx] = {
                    "status": "converged",
                    "bracket": current_bracket,
                    "working_directory": working_directory,
                    "df": df,
                }
                continue

            new_working_directory = _bracket_working_directory(
                working_directory_root, prefix, next_bracket[0], next_bracket[1]
            )
            results[idx] = {
                "status": "resubmitted",
                "bracket": next_bracket,
                "working_directory": new_working_directory,
            }

            iterations_by_idx[idx] += 1
            if iterations_by_idx[idx] >= max_iterations:
                continue  # give up on this row alone; its last submission keeps running

            current_bracket_by_idx[idx] = next_bracket
            new_future, _ = submit_bracket(
                row, next_bracket, calphy_parameters, potential_df,
                executor, working_directory_root, conc_decimals,
            )
            future_to_idx[new_future] = idx

    return pd.DataFrame([results[idx] for idx in structures_df.index], index=structures_df.index)


def _calc_and_maybe_resubmit(
    row: pd.Series,
    calphy_parameters: Dict[str, Any],
    potential_df: pd.DataFrame,
    working_directory: str,
    current_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float],
    step_upper: Optional[float],
    conc_decimals: int,
    working_directory_root: str,
    executor_factory: Callable[[], Any],
    max_iterations: int,
    iterations_used: int,
) -> Tuple[Any, pd.DataFrame]:
    """Run one bracket for real, then chain the next one from *inside* this call.

    This is what actually gets submitted -- not `calc_free_energy_with_calphy`
    directly. It runs the real calphy calculation itself, and if the result
    doesn't converge, builds a fresh executor via `executor_factory()` and
    submits *itself* again for the narrower bracket, right here, before
    returning -- so the chain of resubmissions happens whichever process
    actually executes this function (e.g. a SLURM compute node), not in
    whatever process called `submit_self_resubmitting_bracket` in the first
    place. That process can disconnect immediately after the first
    submission; nothing further is required from it.

    Deliberately checks disk before doing (or redoing) any work, for two
    reasons that both matter here specifically because `tolerance` is one of
    this function's own arguments (unlike `submit_bracket`, which excludes
    it): first, a bracket that's already on disk is reused as-is rather than
    recomputed -- calphy's MD is stochastic, so rerunning an
    already-computed bracket (e.g. because a second, independent call
    bootstrapped back to the same "current" bracket) can silently produce a
    *different* criterion than the first time, which is exactly how a
    bracket that already converged can end up narrowed past: the narrowing
    decision was made from an earlier run's result before this one
    overwrote it. Second, and separately, if some already-tried bracket --
    possibly wider, from an earlier step -- already satisfies `tolerance`,
    there's nothing left to do at all: don't recompute `current_bracket` and
    don't resubmit anything narrower.
    """
    prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
    tried_brackets = _find_tried_brackets(working_directory_root, prefix)
    criteria_by_bracket = _tried_bracket_criteria(prefix, working_directory_root, tried_brackets)

    best_converged = pick_best_converged_bracket(criteria_by_bracket, tolerance)
    if best_converged is not None:
        converged_working_directory = _bracket_working_directory(
            working_directory_root, prefix, best_converged[0], best_converged[1]
        )
        return None, gather_calphy_results_detailed(converged_working_directory)

    # Approximate equality, matching get_unique_criterion_table: t_low/t_high
    # round-trip through the .2f directory-naming format, so a bracket
    # combined or stepped in-memory can differ from the parsed disk value in
    # the last decimal without being a genuinely different bracket.
    already_computed = any(
        np.isclose(bracket[0], current_bracket[0], atol=0.005) and np.isclose(bracket[1], current_bracket[1], atol=0.005)
        for bracket in criteria_by_bracket
    )
    if already_computed:
        # Already computed (just not within tolerance) -- reuse it instead
        # of rerunning calphy and risking a different stochastic result.
        df = gather_calphy_results_detailed(working_directory)
        input_class = None
    else:
        params = dict(calphy_parameters)
        params["temperature"] = list(current_bracket)
        input_class, df = calc_free_energy_with_calphy(
            input_structure=row["atoms"],
            potential_df=potential_df,
            calphy_parameters=params,
            working_directory=working_directory,
        )

    result_row = df.iloc[0]
    forward = result_row["forward_energy_diff"]
    backward = result_row["backward_energy_diff"]

    if forward is None or backward is None:
        return input_class, df  # incomplete (e.g. wrong mode); nothing to chain

    criterion = ts_overlap_criterion(forward[0], backward[0])
    next_bracket = decide_next_bracket(current_bracket, criterion, tolerance, step_lower=step_lower, step_upper=step_upper)

    if next_bracket is None:
        return input_class, df  # converged

    if iterations_used >= max_iterations:
        return input_class, df  # give up on this row; this bracket is the final state

    new_working_directory = _bracket_working_directory(
        working_directory_root, prefix, next_bracket[0], next_bracket[1]
    )

    next_executor = executor_factory()
    next_executor.submit(
        _calc_and_maybe_resubmit,
        row=row,
        calphy_parameters=calphy_parameters,
        potential_df=potential_df,
        working_directory=new_working_directory,
        current_bracket=next_bracket,
        tolerance=tolerance,
        step_lower=step_lower,
        step_upper=step_upper,
        conc_decimals=conc_decimals,
        working_directory_root=working_directory_root,
        executor_factory=executor_factory,
        max_iterations=max_iterations,
        iterations_used=iterations_used + 1,
    )
    next_executor.shutdown(wait=False, cancel_futures=False)

    return input_class, df


def submit_self_resubmitting_bracket(
    row: pd.Series,
    calphy_parameters: Dict[str, Any],
    potential_df: pd.DataFrame,
    executor: Any,
    executor_factory: Callable[[], Any],
    working_directory_root: str,
    initial_bracket: Tuple[float, float],
    tolerance: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    conc_decimals: int = 6,
    max_iterations: int = 10,
) -> Tuple[Any, str]:
    """Submit one row as a self-chaining bracket and return immediately.

    Unlike `auto_refine_temperature_brackets` (which blocks in the calling
    process until every row converges), this submits once and returns right
    away: the submitted job itself (`_calc_and_maybe_resubmit`) decides
    whether to resubmit the next narrower bracket, using `executor_factory`
    to build a fresh executor wherever *it* is running. For a cluster
    executor (e.g. `SlurmClusterExecutor`), that means the whole narrowing
    chain runs as a sequence of independent SLURM jobs with nothing waiting
    on them in the calling process -- submit and walk away. For an executor
    that runs in the same process it was created in (e.g.
    `SingleNodeExecutor`), the calling process still has to stay alive for
    the chain to run, the same as today; this only changes anything for a
    genuinely separate-process/cluster executor.

    Parameters
    ----------
    row : pd.Series
        One row of a structures DataFrame.
    calphy_parameters, potential_df, working_directory_root, initial_bracket,
    tolerance, step_lower, step_upper, conc_decimals, max_iterations :
        Same meaning as in `refine_temperature_bracket`/
        `auto_refine_temperature_brackets`.
    executor : Any
        Executor used for *this* first submission only.
    executor_factory : Callable[[], Any]
        Zero-argument callable that builds a fresh executor equivalent to
        `executor`, called from within the running job each time it needs
        to submit the next bracket. Must be picklable (e.g. a module-level
        function or a lambda closing over only picklable values), since it
        travels with the submitted task to wherever it actually executes.

    Returns
    -------
    Tuple[Any, str]
        The future for the first submission, and its working directory.
        Nothing about later brackets in the chain is returned here -- check
        on them later via `build_criterion_table`, same as manual mode.
    """
    prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
    tried_brackets = _find_tried_brackets(working_directory_root, prefix)
    current_bracket = resolve_current_bracket(initial_bracket, tried_brackets)
    working_directory = _bracket_working_directory(
        working_directory_root, prefix, current_bracket[0], current_bracket[1]
    )

    future = executor.submit(
        _calc_and_maybe_resubmit,
        row=row,
        calphy_parameters=calphy_parameters,
        potential_df=potential_df,
        working_directory=working_directory,
        current_bracket=current_bracket,
        tolerance=tolerance,
        step_lower=step_lower,
        step_upper=step_upper,
        conc_decimals=conc_decimals,
        working_directory_root=working_directory_root,
        executor_factory=executor_factory,
        max_iterations=max_iterations,
        iterations_used=0,
    )

    return future, working_directory


def build_criterion_table(
    structures_df: pd.DataFrame,
    working_directory_root: str,
    conc_decimals: int = 6,
) -> pd.DataFrame:
    """Gather the TI overlap criterion for every bracket attempted so far.

    Direct replacement for jackall's `get_phase_change_criterion_table`, but
    reading from disk (via the naming convention) instead of a pyiron job
    table. Reports the raw criterion value per attempt -- no tolerance is
    applied here, so the same table can be judged against different
    thresholds later without rerunning anything.

    Parameters
    ----------
    structures_df : pd.DataFrame
        Structures DataFrame (e.g. from `generate_random_binary_structures`),
        one row per concentration/phase.
    working_directory_root : str
        Directory under which per-bracket working directories were created.
    conc_decimals : int, optional
        Decimal precision used when formatting `c_in` into the naming prefix;
        must match what was used when submitting.

    Returns
    -------
    pd.DataFrame
        One row per (concentration, phase, attempted bracket), with columns
        `main_element`, `mixing_element`, `phase_type`, `reference_phase`,
        `c_in`, `t_low`, `t_high`, `criterion` (NaN if unavailable), and
        `working_directory`.
    """
    records: List[Dict[str, Any]] = []

    for _, row in structures_df.iterrows():
        prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
        for t_low, t_high in _find_tried_brackets(working_directory_root, prefix):
            working_directory = _bracket_working_directory(working_directory_root, prefix, t_low, t_high)

            criterion = float("nan")
            try:
                result_df = gather_calphy_results_detailed(working_directory)
                result_row = result_df.iloc[0]
                forward = result_row["forward_energy_diff"]
                backward = result_row["backward_energy_diff"]
                if forward is not None and backward is not None:
                    criterion = ts_overlap_criterion(forward[0], backward[0])
            except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError):
                # A job still writing (or with truncated/corrupted) output can
                # fail to parse; treat it the same as "not available yet"
                # rather than crashing the whole table build.
                pass

            records.append({
                "main_element": row["main_element"],
                "mixing_element": row["mixing_element"],
                "phase_type": row["phase_type"],
                "reference_phase": row["reference_phase"],
                "c_in": row["c_in"],
                "t_low": t_low,
                "t_high": t_high,
                "criterion": criterion,
                "working_directory": working_directory,
            })

    return pd.DataFrame(records)


def get_unique_criterion_table(
    criterion_table: pd.DataFrame,
    initial_bracket: Tuple[float, float],
    tolerance: Optional[float] = None,
) -> pd.DataFrame:
    """Collapse a criterion table to one row per concentration.

    Direct replacement for jackall's `get_unique_phase_change_criterion_table`.

    If `tolerance` is given, picks -- per concentration/phase -- the *widest*
    (least-narrowed) tried bracket whose criterion already satisfies that
    tolerance. This matters because narrowing is a one-way ratchet driven by
    whatever tolerance was in effect at the time: if you ran with a strict
    tolerance first and later want to judge against a looser one, the extra
    narrower reruns done to satisfy the strict tolerance were unnecessary for
    the looser one, and picking the narrowest bracket on disk would silently
    prefer them anyway. Passing the current `tolerance` here recovers the
    coarsest (cheapest) bracket that was already good enough, instead of
    discarding that datapoint in favor of a later, unnecessarily narrow one.

    If `tolerance` is omitted (or nothing yet satisfies it for a given
    concentration), falls back to the "current" bracket -- the same
    combined bracket `refine_temperature_bracket` would resubmit next (via
    `resolve_current_bracket`), i.e. the narrowest bracket tried so far.

    Parameters
    ----------
    criterion_table : pd.DataFrame
        Output of `build_criterion_table`.
    initial_bracket : Tuple[float, float]
        The `(t_low, t_high)` bracket each sweep started from.
    tolerance : float, optional
        Overlap-criterion tolerance to judge convergence against. If not
        given, the narrowest tried bracket is reported regardless of whether
        it actually converged.

    Returns
    -------
    pd.DataFrame
        One row per concentration/phase.
    """
    rows = []
    for _, group in criterion_table.groupby(_IDENTITY_COLUMNS):
        if tolerance is not None:
            converged = group[group["criterion"] <= tolerance]
            if not converged.empty:
                widths = converged["t_high"] - converged["t_low"]
                rows.append(converged.loc[[widths.idxmax()]].iloc[0])
                continue

        tried_brackets = list(zip(group["t_low"], group["t_high"]))
        current = resolve_current_bracket(initial_bracket, tried_brackets)
        # Approximate equality: t_low/t_high round-trip through the .2f
        # directory-naming format, so a current bracket combined from an
        # unrounded initial_bracket can differ from the table's parsed
        # values in the last decimal without being a genuinely different bracket.
        match = group[
            np.isclose(group["t_low"], current[0], atol=0.005)
            & np.isclose(group["t_high"], current[1], atol=0.005)
        ]
        if match.empty:
            # Bounds were narrowed independently across attempts, so no single
            # attempt matches the combined bracket -- fall back to the
            # narrowest attempt actually on disk.
            widths = group["t_high"] - group["t_low"]
            match = group.loc[[widths.idxmin()]]
        rows.append(match.iloc[0])

    return pd.DataFrame(rows).reset_index(drop=True)


def plot_criterion_vs_concentration(
    unique_table: pd.DataFrame,
    tolerance: Optional[float] = None,
    color_by: str = "t_high",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the current-bracket criterion against concentration.

    Direct replacement for the `epsilon_unique` scatter-plus-colorbar cell:
    one point per concentration, colored by how narrow its bracket already
    is, with an optional horizontal reference line for the tolerance under
    consideration. Call this repeatedly with different `tolerance` values to
    tune the threshold without recomputing anything.

    Parameters
    ----------
    unique_table : pd.DataFrame
        Output of `get_unique_criterion_table`.
    tolerance : Optional[float], optional
        If given, drawn as a horizontal reference line.
    color_by : str, optional
        Column to color points by, typically `"t_high"` or `"t_low"`
        depending on which bound is being narrowed.
    ax : Optional[plt.Axes], optional
        Axes to draw on; a new figure/axes is created if not given.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
    """
    _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    scatter = ax.scatter(unique_table["c_in"], unique_table["criterion"], c=unique_table[color_by], cmap="viridis")
    fig.colorbar(scatter, ax=ax, label=color_by)

    if tolerance is not None:
        ax.axhline(tolerance, color="k", linestyle="--", label="tolerance")
        ax.legend(frameon=False)

    ax.set_xlabel("Concentration (c)")
    ax.set_ylabel("TI overlap criterion")

    return fig, ax


def plot_criterion_by_bracket(
    criterion_table: pd.DataFrame,
    group_by: str = "t_high",
    tolerance: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay the criterion vs. concentration for every bracket attempted.

    Direct replacement for the repeated "filter by one temperature, plot,
    repeat for the next" cells: groups the full (non-deduplicated) criterion
    table by `group_by` and draws one line per distinct value, so successive
    narrowing steps show up as separate overlaid lines automatically.

    Parameters
    ----------
    criterion_table : pd.DataFrame
        Output of `build_criterion_table` (not the unique/collapsed table).
    group_by : str, optional
        Column to group and label lines by, typically `"t_high"` or `"t_low"`.
    tolerance : Optional[float], optional
        If given, drawn as a horizontal reference line.
    ax : Optional[plt.Axes], optional
        Axes to draw on; a new figure/axes is created if not given.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
    """
    _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for value, group in criterion_table.groupby(group_by):
        group_sorted = group.sort_values("c_in")
        ax.plot(group_sorted["c_in"], group_sorted["criterion"], marker="o", label=f"{group_by}={value:g}")

    if tolerance is not None:
        ax.axhline(tolerance, color="k", linestyle="--", label="tolerance")

    ax.set_xlabel("Concentration (c)")
    ax.set_ylabel("TI overlap criterion")
    ax.legend(frameon=False)

    return fig, ax
