from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

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
    decision, no executor). Stateless, and safe to resubmit unconditionally:
    recomputes the current bracket from whatever brackets have already been
    attempted on disk (see `_find_tried_brackets`), then always calls
    `executor.submit`. With a caching executor (e.g. executorlib's
    `cache_directory`), submitting an already-finished bracket again is
    idempotent -- the cached result is returned without rerunning anything.
    A plain, non-caching executor (e.g. `concurrent.futures.ThreadPoolExecutor`)
    will instead redo the same computation; still correct, just not free.

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
        - "pending": the submitted future hasn't completed yet.
        - "incomplete": the future completed but forward/backward TI data
          isn't available (e.g. still running under the hood, or fe mode).
        - "converged": forward/backward overlap within tolerance.
        - "resubmitted": did not converge; a narrower bracket was submitted.
        Also includes `bracket`, `working_directory`, and (when available)
        `df` and `future`.
    """
    prefix = _bracket_prefix(row, conc_decimals=conc_decimals)
    tried_brackets = _find_tried_brackets(working_directory_root, prefix)
    current_bracket = resolve_current_bracket(initial_bracket, tried_brackets)

    future, working_directory = submit_bracket(
        row, current_bracket, calphy_parameters, potential_df, executor, working_directory_root, conc_decimals
    )

    if not future.done():
        return {
            "status": "pending",
            "bracket": current_bracket,
            "working_directory": working_directory,
            "future": future,
        }

    _, df = future.result()
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

    new_future, new_working_directory = submit_bracket(
        row, next_bracket, calphy_parameters, potential_df, executor, working_directory_root, conc_decimals
    )

    return {
        "status": "resubmitted",
        "bracket": next_bracket,
        "working_directory": new_working_directory,
        "df": df,
        "future": new_future,
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

    Unlike `refine_temperature_bracket`, which never blocks and expects to
    be called again by hand whenever you want to check in, this blocks: it
    repeatedly calls `refine_temperature_bracket` for every row that hasn't
    converged yet, waits for at least one of their jobs to finish before
    checking again, and keeps going until every row is "converged"/
    "incomplete" or `max_iterations` check-in rounds have passed.

    All rows are driven concurrently rather than one at a time -- each
    round submits/reconnects every still-unconverged row before waiting, so
    independent concentrations progress through SLURM in parallel instead
    of blocking on one row's full narrowing history before starting the next.

    `tolerance` here doesn't have to be right: nothing on disk is ever
    destroyed, and every attempted bracket stays inspectable via
    `build_criterion_table` regardless of what this function decided. A
    row that never converges within `max_iterations` just keeps its last
    status (e.g. still "resubmitted", with its narrowest bracket left
    running) -- this does not raise, it simply stops waiting on that row.

    Parameters
    ----------
    structures_df : pd.DataFrame
        Structures DataFrame, one row per concentration/phase.
    calphy_parameters, potential_df, executor, working_directory_root,
    initial_bracket, tolerance, step_lower, step_upper, conc_decimals :
        Passed straight through to `refine_temperature_bracket` for every row.
    max_iterations : int, optional
        Maximum number of check-in rounds across all rows before giving up
        on whichever rows still haven't converged. Guards against both a
        tolerance that's unreachable (endless narrowing) and a job that
        never leaves the SLURM queue (endless waiting on the same future).

    Returns
    -------
    pd.DataFrame
        One row per input row (same index), with the same columns
        `refine_temperature_bracket` returns (`status`, `bracket`,
        `working_directory`, and `df`/`future` where available).
    """
    results: Dict[Any, Dict[str, Any]] = {}
    pending_idx = list(structures_df.index)

    for _ in range(max_iterations):
        if not pending_idx:
            break

        still_pending = []
        futures_this_round = []

        for idx in pending_idx:
            result = refine_temperature_bracket(
                row=structures_df.loc[idx],
                calphy_parameters=calphy_parameters,
                potential_df=potential_df,
                executor=executor,
                working_directory_root=working_directory_root,
                initial_bracket=initial_bracket,
                tolerance=tolerance,
                step_lower=step_lower,
                step_upper=step_upper,
                conc_decimals=conc_decimals,
            )
            results[idx] = result

            if result["status"] in ("converged", "incomplete"):
                continue

            still_pending.append(idx)
            if "future" in result:
                futures_this_round.append(result["future"])

        pending_idx = still_pending

        if pending_idx and futures_this_round:
            wait(futures_this_round, return_when=FIRST_COMPLETED)

    return pd.DataFrame([results[idx] for idx in structures_df.index], index=structures_df.index)


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
            except FileNotFoundError:
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
) -> pd.DataFrame:
    """Collapse a criterion table to one row per concentration: the current bracket.

    Direct replacement for jackall's `get_unique_phase_change_criterion_table`.
    "Current" is defined the same way `refine_temperature_bracket` defines it
    (via `resolve_current_bracket`), so this always matches what would
    actually be resubmitted next.

    Parameters
    ----------
    criterion_table : pd.DataFrame
        Output of `build_criterion_table`.
    initial_bracket : Tuple[float, float]
        The `(t_low, t_high)` bracket each sweep started from.

    Returns
    -------
    pd.DataFrame
        One row per concentration/phase, matching the current bracket.
    """
    rows = []
    for _, group in criterion_table.groupby(_IDENTITY_COLUMNS):
        tried_brackets = list(zip(group["t_low"], group["t_high"]))
        current = resolve_current_bracket(initial_bracket, tried_brackets)
        match = group[(group["t_low"] == current[0]) & (group["t_high"] == current[1])]
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
