from __future__ import annotations

import os
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
    check_ts_overlap,
    resolve_current_bracket,
    step_bracket,
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

    Stateless, and safe to resubmit unconditionally: recomputes the current
    bracket from whatever brackets have already been attempted on disk (see
    `_find_tried_brackets`), then always calls `executor.submit`. With a
    caching executor (e.g. executorlib's `cache_directory`), submitting an
    already-finished bracket again is idempotent -- the cached result is
    returned without rerunning anything. A plain, non-caching executor (e.g.
    `concurrent.futures.ThreadPoolExecutor`) will instead redo the same
    computation; still correct, just not free.

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
        Passed to `check_ts_overlap` to decide convergence.
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
    t_low, t_high = resolve_current_bracket(initial_bracket, tried_brackets)
    working_directory = _bracket_working_directory(working_directory_root, prefix, t_low, t_high)

    params = dict(calphy_parameters)
    params["temperature"] = [t_low, t_high]

    future = executor.submit(
        calc_free_energy_with_calphy,
        input_structure=row["atoms"],
        potential_df=potential_df,
        calphy_parameters=params,
        working_directory=working_directory,
    )

    if not future.done():
        return {
            "status": "pending",
            "bracket": (t_low, t_high),
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
            "bracket": (t_low, t_high),
            "working_directory": working_directory,
            "df": df,
        }

    if check_ts_overlap(forward[0], backward[0], tolerance):
        return {
            "status": "converged",
            "bracket": (t_low, t_high),
            "working_directory": working_directory,
            "df": df,
        }

    new_t_low, new_t_high = step_bracket(t_low, t_high, step_lower=step_lower, step_upper=step_upper)
    new_working_directory = _bracket_working_directory(working_directory_root, prefix, new_t_low, new_t_high)

    new_future = executor.submit(
        calc_free_energy_with_calphy,
        input_structure=row["atoms"],
        potential_df=potential_df,
        calphy_parameters={**calphy_parameters, "temperature": [new_t_low, new_t_high]},
        working_directory=new_working_directory,
    )

    return {
        "status": "resubmitted",
        "bracket": (new_t_low, new_t_high),
        "working_directory": new_working_directory,
        "df": df,
        "future": new_future,
    }


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
