from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from scipy.spatial import ConvexHull

from phase_diagram_workflows.utils.nested_batch import run_nested_batch

# The convex hull/common-tangent construction (black dotted line, black
# vertex markers) matches landau.plot.plot_excess_free_energy exactly --
# black carries the geometric construction, never data identity. Every
# other structure is small and light gray -- recessive context -- so the
# hull (and its accent-colored labels) reads as the figure, not the full
# point cloud.
_HULL_COLOR = "black"
_OFF_HULL_COLOR = "#b0afaa"


def _structure_color() -> str:
    """The single accent color for hull-point labels, read live from
    seaborn's 'muted' palette (slot 0) -- the same palette landau.plot uses.
    """
    import seaborn as sns

    return sns.color_palette("muted").as_hex()[0]

# -----------------------------------------------------------------------
# Per-structure energy calculation (atomistics lib calculator, LAMMPS)
# -----------------------------------------------------------------------


def optimize_structure(atoms: Atoms, potential_df: pd.DataFrame, **kwargs: Any) -> Atoms:
    """Relax atomic positions and cell volume with a LAMMPS potential.

    Thin wrapper around atomistics' lib-mode LAMMPS calculator, which runs
    LAMMPS in-process (no job/queue overhead of its own) -- suited to being
    submitted over many structures inside an executor (see
    `compute_energies_with_nested_executor`).

    Parameters
    ----------
    atoms : Atoms
        Structure to relax.
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format (Config, Species
        columns).
    **kwargs
        Forwarded to `optimize_positions_and_volume_with_lammpslib` (e.g.
        min_style, etol, ftol, maxiter).

    Returns
    -------
    Atoms
        Relaxed structure.
    """
    from atomistics.calculators.lammps.libcalculator import (
        optimize_positions_and_volume_with_lammpslib,
    )

    return optimize_positions_and_volume_with_lammpslib(
        structure=atoms,
        potential_dataframe=potential_df,
        **kwargs,
    )


def compute_energy_per_atom(atoms: Atoms, potential_df: pd.DataFrame, **kwargs: Any) -> float:
    """Compute the potential energy per atom with a LAMMPS potential.

    Parameters
    ----------
    atoms : Atoms
        Structure to evaluate.
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format.
    **kwargs
        Forwarded to `calc_static_with_lammpslib`.

    Returns
    -------
    float
        Potential energy divided by the number of atoms.
    """
    from atomistics.calculators.lammps.libcalculator import calc_static_with_lammpslib

    result = calc_static_with_lammpslib(
        structure=atoms,
        potential_dataframe=potential_df,
        output_keys=("energy",),
        **kwargs,
    )
    return float(result["energy"]) / len(atoms)


def optimize_and_compute_energy_per_atom(
    atoms: Atoms,
    potential_df: pd.DataFrame,
    optimize_kwargs: Optional[Dict[str, Any]] = None,
    static_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Atoms, float]:
    """Relax a structure, then evaluate its energy per atom at the relaxed geometry.

    This is the per-structure unit of work submitted to executors by
    `compute_energies_with_nested_executor`: one LAMMPS minimization
    followed by one static evaluation, both via atomistics' in-process lib
    calculator.

    Parameters
    ----------
    atoms : Atoms
        Structure to relax and evaluate.
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format.
    optimize_kwargs : Optional[Dict[str, Any]]
        Extra keyword arguments for `optimize_structure`.
    static_kwargs : Optional[Dict[str, Any]]
        Extra keyword arguments for `compute_energy_per_atom`.

    Returns
    -------
    Tuple[Atoms, float]
        The relaxed structure and its energy per atom.
    """
    relaxed = optimize_structure(atoms, potential_df, **(optimize_kwargs or {}))
    energy_per_atom = compute_energy_per_atom(relaxed, potential_df, **(static_kwargs or {}))
    return relaxed, energy_per_atom


def compute_energies_with_nested_executor(
    df: pd.DataFrame,
    atoms_col: str,
    potential_df: pd.DataFrame,
    outer_executor_cls: type,
    task_fn=optimize_and_compute_energy_per_atom,
    result_atoms_col: str = "atoms_relaxed",
    result_value_col: str = "energy_per_atom_calc",
    **run_nested_batch_kwargs: Any,
) -> Any:
    """Compute a per-structure energy for every row of `df` via nested executors.

    Convenience wrapper around `execution.nested_batch.run_nested_batch`,
    specialized to structures dataframes: builds the item list from
    `df[atoms_col]`, passes `potential_df` as a shared extra argument to
    `task_fn`, and (when `wait=True`) attaches the `(relaxed_atoms, value)`
    results back onto `df` as `result_atoms_col`/`result_value_col`.
    `result_value_col` defaults to 'energy_per_atom_calc' rather than
    'energy_per_atom' so it doesn't collide with Materials Project's own DFT
    `energy_per_atom` field when `df` comes from
    `structures.materials_project.build_structures_dataframe`.

    Parameters
    ----------
    df : pd.DataFrame
        Structures dataframe, e.g. from
        `structures.materials_project.build_structures_dataframe`.
    atoms_col : str
        Name of the column holding ASE Atoms objects.
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format, applied to every
        structure.
    outer_executor_cls : type
        executorlib executor class for the outer allocation, e.g.
        `executorlib.SlurmClusterExecutor`.
    task_fn : Callable[[Atoms, pd.DataFrame], Tuple[Atoms, float]]
        Called as `task_fn(atoms, potential_df)` inside the inner executor.
        Must return a `(relaxed_atoms, value)` tuple. Defaults to
        `optimize_and_compute_energy_per_atom`.
    result_atoms_col, result_value_col : str
        Column names the `(relaxed_atoms, value)` results are written to.
    **run_nested_batch_kwargs
        Forwarded to `run_nested_batch` (outer_resource_dict,
        inner_resource_dict, inner_max_workers, cache_directory,
        pysqa_config_directory, wait, and any executor-specific kwargs).

    Returns
    -------
    pd.DataFrame or concurrent.futures.Future
        `df` with results filled in if `wait=True` (the default); otherwise
        the `Future` wrapping the pending batch (see `run_nested_batch`).
    """
    results = run_nested_batch(
        items=df[atoms_col].tolist(),
        task_fn=task_fn,
        outer_executor_cls=outer_executor_cls,
        task_args=(potential_df,),
        **run_nested_batch_kwargs,
    )

    if not run_nested_batch_kwargs.get("wait", True):
        return results  # a Future; caller retrieves results later

    out = df.copy()
    relaxed, values = zip(*results)
    out[result_atoms_col] = list(relaxed)
    out[result_value_col] = list(values)
    return out


# -----------------------------------------------------------------------
# Mixing energy and convex hull
# -----------------------------------------------------------------------


def compute_mixing_energy(
    df: pd.DataFrame,
    x_col: str = "x",
    energy_col: str = "energy_per_atom",
    mixing_energy_col: str = "mixing_energy",
) -> pd.DataFrame:
    """Compute mixing energy relative to the lowest-energy pure endpoints.

    For a binary system with composition axis `x_col` in [0, 1], the mixing
    energy of a structure is its energy minus the linear interpolation
    between the lowest-energy structure at x=0 and the lowest-energy
    structure at x=1::

        E_mix(x) = E(x) - [(1 - x) * E(x=0) + x * E(x=1)]

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `x_col` and `energy_col`.
    x_col : str
        Composition column, in [0, 1].
    energy_col : str
        Per-atom energy column.
    mixing_energy_col : str
        Name of the new column to add.

    Returns
    -------
    pd.DataFrame
        `df` with `mixing_energy_col` added.

    Raises
    ------
    ValueError
        If no rows exist at x=0 or x=1, the endpoints the mixing energy is
        anchored to.
    """
    pure_low = df[df[x_col] == 0]
    pure_high = df[df[x_col] == 1]
    if pure_low.empty:
        raise ValueError(f"No structures found with {x_col} == 0 to anchor the mixing energy.")
    if pure_high.empty:
        raise ValueError(f"No structures found with {x_col} == 1 to anchor the mixing energy.")

    e_low = pure_low[energy_col].min()
    e_high = pure_high[energy_col].min()

    out = df.copy()
    out[mixing_energy_col] = out[energy_col] - ((1 - out[x_col]) * e_low + out[x_col] * e_high)
    return out


def compute_convex_hull(
    df: pd.DataFrame,
    x_col: str = "x",
    energy_col: str = "mixing_energy",
) -> pd.DataFrame:
    """Return the subset of `df` that lies on the lower convex hull.

    The lower hull is the part of the boundary that minimizes `energy_col`
    for each `x_col` -- i.e. the set of stable/ground-state compositions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `x_col` and `energy_col`.
    x_col : str
        Composition column, in [0, 1].
    energy_col : str
        Energy column (typically mixing energy) the hull is computed over.

    Returns
    -------
    pd.DataFrame
        Rows of `df` on the lower hull, sorted by `x_col`.

    Raises
    ------
    ValueError
        If fewer than 3 rows have finite `x_col`/`energy_col` values (a hull
        needs at least 3 points).
    """
    df_finite = df.dropna(subset=[x_col, energy_col])
    if len(df_finite) < 3:
        raise ValueError(
            f"Need at least 3 structures with finite {x_col}/{energy_col} to compute a convex hull, "
            f"got {len(df_finite)}."
        )

    points = df_finite[[x_col, energy_col]].to_numpy()
    hull = ConvexHull(points)

    # A hull edge belongs to the lower envelope when its outward normal
    # points downward (negative energy-axis component): the hull interior
    # lies above such an edge, which is exactly the boundary that minimizes
    # energy_col at each x_col.
    lower_mask = np.zeros(len(df_finite), dtype=bool)
    for simplex, equation in zip(hull.simplices, hull.equations):
        if equation[1] < 0:
            lower_mask[simplex] = True

    df_hull = df_finite[lower_mask].sort_values(x_col).reset_index(drop=True)
    return df_hull


def _default_yrange(
    df_hull: pd.DataFrame,
    energy_col: str,
    max_energy_above_hull: Optional[float],
) -> Optional[Tuple[float, float]]:
    """Y-axis range that always contains the hull, capped above by a fixed
    "energy above hull" window rather than the full data range.

    Structures that failed to relax well can land far above the hull (many
    tenths of an eV/atom) and, left to plain auto-scaling, squash the hull
    itself -- the actually interesting part -- into a thin band at the
    bottom. Capping at `hull_max + max_energy_above_hull` (a standard
    quantity in this kind of plot: MP-style e-above-hull views are
    conventionally windowed the same way, typically 0.1-0.3 eV/atom) keeps
    near-hull context visible and lets those outliers fall outside the view
    instead of dominating it. Returns None (auto-scale) if
    `max_energy_above_hull` is None.
    """
    if max_energy_above_hull is None:
        return None
    hull_lo, hull_hi = df_hull[energy_col].min(), df_hull[energy_col].max()
    pad = max(0.1 * (hull_hi - hull_lo), 1e-3)
    return (hull_lo - pad, hull_hi + max_energy_above_hull)


def _default_axis_labels(
    x_col: str, energy_col: str, element: Optional[str] = None, latex: bool = True
) -> Tuple[str, str]:
    r"""Physical axis labels with units, derived from the (generic) column names.

    The calculator layer in this module (`compute_energy_per_atom` et al.) is
    always eV/atom, so the y-axis unit is never actually ambiguous even
    though `energy_col` is just a column name like 'mixing_energy'. If
    `element` is given (the element `x_col` is the fraction of, e.g. "Mg"
    for `build_structures_dataframe(..., elements=["Al", "Mg"])`), the
    x-label becomes an element-subscripted label matching `landau.plot`'s
    convention (`rf"$c_\mathrm{{{element}}}$"`); otherwise it falls back to
    a generic label. Pass `xlabel`/`ylabel` explicitly to override either
    default outright.

    `latex=True` (matplotlib) gives mathtext (`$x_\mathrm{Mg}$`), which
    matplotlib renders natively. `latex=False` (plotly) gives an HTML
    subscript (`x<sub>Mg</sub>`) instead: plotly's own LaTeX support needs
    MathJax loaded and is off by default in most renderers, so a `$...$`
    label there just shows up as literal, unrendered text.
    """
    if element:
        xlabel = rf"$x_\mathrm{{{element}}}$" if latex else f"x<sub>{element}</sub>"
    else:
        xlabel = "Composition, $x$" if latex else "Composition, x"
    ylabel = "Mixing energy [eV/atom]" if "mixing" in energy_col.lower() else "Energy per atom [eV/atom]"
    return xlabel, ylabel


def plot_convex_hull(
    df: pd.DataFrame,
    x_col: str = "x",
    energy_col: str = "mixing_energy",
    label_col: Optional[str] = "formula_pretty",
    color_col: Optional[str] = None,
    hover_cols: Optional[Tuple[str, ...]] = None,
    fig_width: int = 800,
    fig_height: int = 550,
    yrange: Optional[Tuple[float, float]] = None,
    max_energy_above_hull: Optional[float] = 0.2,
    element: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    r"""Plot mixing energy vs. composition, with the convex hull overlaid.

    Styled like `landau.plot.plot_excess_free_energy`: structures are
    colored points (seaborn "muted" slot 0 -- color carries data identity),
    the convex hull is a black dotted line with black vertex markers (our
    equivalent of landau's common-tangent construction -- black carries the
    geometric construction, never data identity). Coloring every point by a
    high-cardinality column like space group (pass `color_col` to opt back
    into that) reads as visual noise for tens of structures and isn't the
    scientifically interesting dimension here -- being on the hull is.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `x_col` and `energy_col`.
    x_col : str
        Composition column, in [0, 1].
    energy_col : str
        Energy column (typically mixing energy) to plot on the y-axis.
    label_col : Optional[str]
        Column to use for direct text labels on hull points (e.g.
        'formula_pretty'). Set to None to omit labels.
    color_col : Optional[str]
        If given, color *all* points by this column instead of the default
        gray/accent status styling -- useful for ad hoc exploration (e.g.
        'space_group'), not recommended for a final figure with many groups.
    hover_cols : Optional[Tuple[str, ...]]
        Columns to show on hover. Defaults to scalar-typed columns of `df`.
    fig_width, fig_height : int
        Figure size in pixels.
    yrange : Optional[Tuple[float, float]]
        Y-axis range as (min, max). Overrides `max_energy_above_hull`.
    max_energy_above_hull : Optional[float]
        When `yrange` is not given, caps the y-axis at `hull_max +
        max_energy_above_hull` instead of the full data range, so a few
        badly-relaxed outliers can't squash the hull into a thin band at
        the bottom (see `_default_yrange`). None auto-scales to all data.
    element : Optional[str]
        The element `x_col` is the fraction of (e.g. "Mg"). If given, the
        default x-label becomes the LaTeX `$x_\mathrm{Mg}$` (see
        `_default_axis_labels`).
    xlabel, ylabel : Optional[str]
        Axis labels. Default to an `element`-subscripted or generic
        composition label and an eV/atom energy label inferred from
        `energy_col` (see `_default_axis_labels`) -- override either
        outright.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive scatter plot with the lower convex hull overlaid.
    df_hull : pd.DataFrame
        Rows of `df` on the lower hull (see `compute_convex_hull`).
    """
    import warnings

    import plotly.express as px
    import plotly.graph_objects as go

    df_plot = df.dropna(subset=[x_col, energy_col]).copy()
    df_hull = compute_convex_hull(df_plot, x_col=x_col, energy_col=energy_col)

    if yrange is not None:
        lo, hi = yrange
        clipped = df_hull[(df_hull[energy_col] < lo) | (df_hull[energy_col] > hi)]
        if not clipped.empty:
            warnings.warn(
                f"yrange={yrange} clips {len(clipped)} hull point(s) with "
                f"{energy_col} outside that range (e.g. {clipped[energy_col].iloc[0]:.4g}); "
                "pass a wider yrange or None to auto-scale.",
                stacklevel=2,
            )
    else:
        yrange = _default_yrange(df_hull, energy_col, max_energy_above_hull)

    default_xlabel, default_ylabel = _default_axis_labels(x_col, energy_col, element=element, latex=False)
    xlabel = xlabel or default_xlabel
    ylabel = ylabel or default_ylabel

    if hover_cols is None:
        hover_cols = tuple(
            col
            for col in df_plot.columns
            if not df_plot[col].dropna().empty
            and isinstance(df_plot[col].dropna().iloc[0], (int, float, str, bool, np.integer, np.floating))
        )

    hovertemplate = "".join(f"{col}: %{{customdata[{i}]}}<br>" for i, col in enumerate(hover_cols))
    hovertemplate += "<extra></extra>"

    fig = go.Figure()

    if color_col is not None:
        groups = df_plot[color_col].unique().tolist()
        palette = (
            px.colors.qualitative.Plotly
            if len(groups) <= len(px.colors.qualitative.Plotly)
            else px.colors.qualitative.Alphabet
        )
        for i, group in enumerate(groups):
            group_df = df_plot[df_plot[color_col] == group]
            group_customdata = group_df[list(hover_cols)].to_numpy() if hover_cols else None
            fig.add_trace(
                go.Scatter(
                    x=group_df[x_col],
                    y=group_df[energy_col],
                    mode="markers",
                    name=str(group),
                    marker=dict(size=9, color=palette[i % len(palette)]),
                    customdata=group_customdata,
                    hovertemplate=hovertemplate,
                )
            )
    else:
        customdata = df_plot[list(hover_cols)].to_numpy() if hover_cols else None
        fig.add_trace(
            go.Scatter(
                x=df_plot[x_col],
                y=df_plot[energy_col],
                mode="markers",
                name="Structures",
                marker=dict(size=6, color=_OFF_HULL_COLOR),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_hull[x_col],
            y=df_hull[energy_col],
            mode="lines+markers" + ("+text" if label_col else ""),
            name="Convex hull",
            text=df_hull[label_col] if label_col else None,
            textposition="top center",
            textfont=dict(size=12, color=_structure_color()),
            line=dict(color=_HULL_COLOR, width=1.5, dash="dot"),
            marker=dict(size=7, color=_HULL_COLOR),
        )
    )

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=fig_width,
        height=fig_height,
        yaxis_range=yrange,
        template="simple_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=30, l=70, b=60),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

    return fig, df_hull


def plot_convex_hull_matplotlib(
    df: pd.DataFrame,
    x_col: str = "x",
    energy_col: str = "mixing_energy",
    label_col: Optional[str] = "formula_pretty",
    yrange: Optional[Tuple[float, float]] = None,
    max_energy_above_hull: Optional[float] = 0.2,
    element: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (6.0, 4.5),
    ax=None,
):
    r"""Static, publication-style convex hull plot (matplotlib).

    Same landau-matching styling as `plot_convex_hull` (colored structures,
    black dotted hull -- see its docstring), but static and minimal: no
    legend clutter, no gridlines, top/right spines removed -- meant to be
    saved directly as a figure (`fig.savefig(..., dpi=300)`) rather than
    explored interactively.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `x_col` and `energy_col`.
    x_col, energy_col : str
        Composition and energy columns.
    label_col : Optional[str]
        Column to label hull points with (e.g. 'formula_pretty'). None to
        omit labels.
    yrange : Optional[Tuple[float, float]]
        Y-axis range as (min, max). Overrides `max_energy_above_hull`.
    max_energy_above_hull : Optional[float]
        When `yrange` is not given, caps the y-axis at `hull_max +
        max_energy_above_hull` instead of the full data range, so a few
        badly-relaxed outliers can't squash the hull into a thin band at
        the bottom (see `_default_yrange`). None auto-scales to all data.
    element : Optional[str]
        The element `x_col` is the fraction of (e.g. "Mg"). If given, the
        default x-label becomes the LaTeX `$x_\mathrm{Mg}$` (see
        `_default_axis_labels`).
    xlabel, ylabel : Optional[str]
        Axis labels. Default to an `element`-subscripted or generic
        composition label and an eV/atom energy label inferred from
        `energy_col` (see `_default_axis_labels`) -- override either
        outright.
    figsize : Tuple[float, float]
        Figure size in inches, used only if `ax` is None.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure/axes is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    df_hull : pd.DataFrame
        Rows of `df` on the lower hull (see `compute_convex_hull`).
    """
    import warnings

    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt

    df_plot = df.dropna(subset=[x_col, energy_col]).copy()
    df_hull = compute_convex_hull(df_plot, x_col=x_col, energy_col=energy_col)

    if yrange is not None:
        lo, hi = yrange
        clipped = df_hull[(df_hull[energy_col] < lo) | (df_hull[energy_col] > hi)]
        if not clipped.empty:
            warnings.warn(
                f"yrange={yrange} clips {len(clipped)} hull point(s) with "
                f"{energy_col} outside that range (e.g. {clipped[energy_col].iloc[0]:.4g}); "
                "pass a wider yrange or None to auto-scale.",
                stacklevel=2,
            )
    else:
        yrange = _default_yrange(df_hull, energy_col, max_energy_above_hull)

    default_xlabel, default_ylabel = _default_axis_labels(x_col, energy_col, element=element, latex=True)
    xlabel = xlabel or default_xlabel
    ylabel = ylabel or default_ylabel

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    structure_color = _structure_color()
    # Landau's common-tangent styling for the hull (see plot_excess_free_energy):
    # black dotted line, lw=1.5, zorder=3; black hull-vertex dots, s=25,
    # zorder=7; labels via _text_with_outline's own defaults (fontsize
    # "small", bold, 3px white stroke, zorder=10). Off-hull structures are
    # small, light gray, recessive context -- not part of landau's scheme
    # (which has no "off the hull" cloud to de-emphasize).
    ax.scatter(
        df_plot[x_col], df_plot[energy_col],
        color=_OFF_HULL_COLOR, s=18, zorder=2,
    )
    ax.plot(
        df_hull[x_col], df_hull[energy_col],
        linestyle="dotted", color=_HULL_COLOR, linewidth=1.5, zorder=3,
    )
    ax.scatter(
        df_hull[x_col], df_hull[energy_col],
        color=_HULL_COLOR, s=25, zorder=7,
    )

    if label_col:
        outline = [patheffects.withStroke(linewidth=3, foreground="white")]
        for _, row in df_hull.iterrows():
            ax.annotate(
                str(row[label_col]),
                xy=(row[x_col], row[energy_col]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize="small",
                fontweight="bold",
                color=structure_color,
                path_effects=outline,
                zorder=10,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yrange:
        ax.set_ylim(*yrange)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color=".5", linestyle="--", zorder=1)
    fig.tight_layout()

    return fig, ax, df_hull


# -----------------------------------------------------------------------
# High-level, two-step workflow: fetch+compute, then analyze+plot
# -----------------------------------------------------------------------


def fetch_structures_and_energies(
    elements: Sequence[str],
    api_key: str,
    potential_df: pd.DataFrame,
    outer_executor_cls: type,
    extra_structures: Optional[List[Dict[str, Any]]] = None,
    include_pure: bool = True,
    fields: Optional[List[str]] = None,
    task_fn=optimize_and_compute_energy_per_atom,
    **run_nested_batch_kwargs: Any,
) -> pd.DataFrame:
    """Step 1: fetch structures from Materials Project and compute their energies.

    One call from "a chemical system and a potential" to "a dataframe with a
    relaxed structure and energy per atom for every entry": fetches the
    chemical system spanned by `elements` (plus its pure elements) from
    Materials Project, builds the structures dataframe, optionally folds in
    structures of your own (e.g. a hand-built supercell not on Materials
    Project), and computes energies for all of them via a nested executor.
    Feed the result to `analyze_convex_hull` for step 2.

    Parameters
    ----------
    elements : Sequence[str]
        The two elements defining the system, e.g. ["Al", "Mg"] -- both the
        Materials Project chemical system to fetch ("Al-Mg") and the
        composition axis (fraction of `elements[1]`).
    api_key : str
        Materials Project API key.
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format, applied to every
        structure.
    outer_executor_cls : type
        executorlib executor class for the outer allocation, e.g.
        `executorlib.SlurmClusterExecutor`.
    extra_structures : Optional[List[Dict[str, Any]]]
        Structures not on Materials Project to fold in, e.g. `[{"atoms":
        beta_atoms, "material_id": "external-beta", "formula_pretty":
        "Al534Mg345"}]`. Each dict is passed as keyword arguments to
        `structures.materials_project.append_structure`.
    include_pure, fields
        Forwarded to `structures.materials_project.get_materials_project_df`.
    task_fn
        Forwarded to `compute_energies_with_nested_executor`.
    **run_nested_batch_kwargs
        Forwarded to `compute_energies_with_nested_executor` /
        `run_nested_batch` (outer_resource_dict, inner_resource_dict,
        inner_max_workers, cache_directory, pysqa_config_directory, wait,
        ...).

    Returns
    -------
    pd.DataFrame
        One row per structure, with `structure_ase`, `x`, `atoms_relaxed`,
        and `energy_per_atom_calc` columns (see
        `structures.materials_project.build_structures_dataframe` and
        `compute_energies_with_nested_executor`).
    """
    from phase_diagram_workflows.structures.materials_project import (
        append_structure,
        build_structures_dataframe,
        get_materials_project_df,
    )

    chemsys = "-".join(elements)
    mp_df = get_materials_project_df(chemsys, api_key, include_pure=include_pure, fields=fields)
    df = build_structures_dataframe(mp_df, elements=elements)

    for extra in extra_structures or []:
        df = append_structure(df, elements=elements, **extra)

    return compute_energies_with_nested_executor(
        df=df,
        atoms_col="structure_ase",
        potential_df=potential_df,
        outer_executor_cls=outer_executor_cls,
        task_fn=task_fn,
        **run_nested_batch_kwargs,
    )


def analyze_convex_hull(
    df: pd.DataFrame,
    x_col: str = "x",
    energy_col: str = "energy_per_atom_calc",
    mixing_energy_col: str = "mixing_energy",
    label_col: Optional[str] = "formula_pretty",
    backend: str = "plotly",
    **plot_kwargs: Any,
):
    """Step 2: mixing energy, convex hull, and a ready-to-show plot, in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `fetch_structures_and_energies` (or any dataframe with
        `x_col` and `energy_col`).
    x_col, energy_col : str
        Composition and per-atom energy columns.
    mixing_energy_col : str
        Name for the computed mixing-energy column (see
        `compute_mixing_energy`).
    label_col : Optional[str]
        Column to label hull points with. None to omit labels.
    backend : str
        'plotly' (interactive, `plot_convex_hull`) or 'matplotlib' (static,
        publication-style, `plot_convex_hull_matplotlib`).
    **plot_kwargs
        Forwarded to the chosen plotting function.

    Returns
    -------
    For backend='plotly': `(fig, df_hull)` (see `plot_convex_hull`).
    For backend='matplotlib': `(fig, ax, df_hull)` (see
    `plot_convex_hull_matplotlib`).
    """
    df_mix = compute_mixing_energy(df, x_col=x_col, energy_col=energy_col, mixing_energy_col=mixing_energy_col)

    if backend == "plotly":
        return plot_convex_hull(
            df_mix, x_col=x_col, energy_col=mixing_energy_col, label_col=label_col, **plot_kwargs
        )
    elif backend == "matplotlib":
        return plot_convex_hull_matplotlib(
            df_mix, x_col=x_col, energy_col=mixing_energy_col, label_col=label_col, **plot_kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'plotly' or 'matplotlib'.")
