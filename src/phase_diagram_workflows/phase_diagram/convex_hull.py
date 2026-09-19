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


def _structure_color(index: int = 0) -> str:
    """An accent color for hull-point labels, read live from seaborn's
    'muted' palette -- the same palette landau.plot uses. Slot 0 (the
    default) is the primary series; a DFT overlay (see `dft_mixing_energy_col`
    on the plot functions) uses slot 1 so the two series stay visually
    distinct but drawn from the same consistent palette.
    """
    import seaborn as sns

    return sns.color_palette("muted").as_hex()[index]


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _subscript_digits(text: str) -> str:
    """Render a chemical formula's atom counts as subscripts, e.g.
    'Mg23Al30' -> 'Mg₂₃Al₃₀'.

    Uses plain Unicode subscript characters rather than matplotlib mathtext
    (`$_{23}$`): mathtext renders in a different font to the surrounding
    bold label text, so mixing the two within one string produces a visibly
    inconsistent, clashing look. Unicode subscripts stay in the same font
    as everything else.
    """
    return text.translate(_SUBSCRIPT_DIGITS)


def _place_labels_no_overlap(
    fig,
    ax,
    entries: List[Dict[str, Any]],
    base_offset: float = 9.0,
    step: float = 9.0,
    max_rings: int = 20,
):
    """Annotate points with text labels, placing each to avoid overlap.

    For each label (processed left-to-right by x), tries a sequence of
    candidate vertical offsets -- above the point first, then below, then
    further above, then further below, and so on outward -- taking the
    first one whose text doesn't overlap any label already placed. Most
    labels land at the first (closest, no-nudge) candidate on one side or
    the other; only genuinely crowded spots need to go further out. A
    label only gets a leader line back to its point when it had to go
    beyond the immediate above/below pair -- the common case (an
    uncontested spot right above or right below the point) needs no line,
    which is exactly what a plain `ax.annotate` would have done, just also
    trying below instead of only ever pushing labels further above.

    Processing points in a fixed left-to-right order and never re-touching
    an already-placed label means each label's search only depends on
    labels already resolved -- unlike resolving arbitrary overlapping pairs
    against each other, which can oscillate two labels back into the same
    offset instead of converging.

    Parameters
    ----------
    fig, ax : matplotlib Figure/Axes
        Must belong to a canvas that supports `get_renderer()` (true for the
        default Agg/interactive backends).
    entries : List[Dict[str, Any]]
        One dict per label, with keys 'x', 'y', 'text', 'color', and
        optionally 'path_effects' (defaults to a white outline stroke).
    base_offset : float
        Vertical offset in points of the closest above/below candidates.
    step : float
        How far out (in points) each successive ring of candidates goes.
    max_rings : int
        Safety cap on how many above/below rings to try for any single
        label, for pathological inputs; typical hull sizes (a handful of
        points, moderate overlap) resolve within the first ring or two.

    Returns
    -------
    List[matplotlib.text.Annotation]
        The final annotation objects, in `entries` order (not sweep order).
    """
    import matplotlib.patheffects as patheffects

    default_outline = [patheffects.withStroke(linewidth=3, foreground="white")]

    def draw(entry: Dict[str, Any], offset: float) -> Any:
        return ax.annotate(
            entry["text"],
            xy=(entry["x"], entry["y"]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize="small",
            fontweight="bold",
            color=entry["color"],
            path_effects=entry.get("path_effects", default_outline),
            zorder=10,
        )

    def candidate_offsets():
        for ring in range(max_rings):
            yield base_offset + ring * step
            yield -(base_offset + ring * step)

    if not entries:
        return []

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    annotations: List[Any] = [None] * len(entries)
    offsets = [base_offset] * len(entries)
    placed_bboxes: List[Any] = []

    for i in sorted(range(len(entries)), key=lambda idx: entries[idx]["x"]):
        for offset in candidate_offsets():
            ann = draw(entries[i], offset)
            bbox = ann.get_window_extent(renderer)
            if not any(bbox.overlaps(placed) for placed in placed_bboxes):
                break
            ann.remove()
        annotations[i] = ann
        offsets[i] = offset
        placed_bboxes.append(bbox)

    # A label only gets a leader line if it needed a ring beyond the
    # immediate above/below pair -- keeps the common case (an uncontested
    # spot right above or below) identical to a plain, line-free annotate.
    for i, entry in enumerate(entries):
        if abs(abs(offsets[i]) - base_offset) > 1e-9:
            annotations[i].remove()
            annotations[i] = ax.annotate(
                entry["text"],
                xy=(entry["x"], entry["y"]),
                xytext=(0, offsets[i]),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize="small",
                fontweight="bold",
                color=entry["color"],
                path_effects=entry.get("path_effects", default_outline),
                zorder=10,
                arrowprops=dict(arrowstyle="-", color=entry["color"], lw=0.6, alpha=0.6, shrinkA=1, shrinkB=3),
            )

    return annotations

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
    dft_mixing_energy_col: Optional[str] = None,
    hover_cols: Optional[Tuple[str, ...]] = None,
    fig_width: int = 800,
    fig_height: int = 550,
    yrange: Optional[Tuple[float, float]] = None,
    max_energy_above_hull: Optional[float] = None,
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
    dft_mixing_energy_col : Optional[str]
        Column holding a second, already-computed mixing energy to overlay
        as a comparison hull trace -- e.g. one built from Materials
        Project's own `energy_per_atom` (see `compute_mixing_energy`),
        alongside the primary `energy_col`. Drawn as a dashed line with
        square markers, labeled "Convex hull (DFT)". None (default) omits
        it entirely.
    hover_cols : Optional[Tuple[str, ...]]
        Columns to show on hover. Defaults to scalar-typed columns of `df`.
    fig_width, fig_height : int
        Figure size in pixels.
    yrange : Optional[Tuple[float, float]]
        Y-axis range as (min, max). Overrides `max_energy_above_hull`.
    max_energy_above_hull : Optional[float]
        None (default): auto-scale the y-axis to all data. When `yrange`
        is not given and this is a float instead, caps the y-axis at
        `hull_max + max_energy_above_hull` so a few badly-relaxed outliers
        can't squash the hull into a thin band at the bottom (see
        `_default_yrange`) -- opt into this if auto-scaling looks bad for
        your data, rather than it being the default.
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

    def warn_if_clips(hull, col, label):
        if yrange is None:
            return
        lo, hi = yrange
        clipped = hull[(hull[col] < lo) | (hull[col] > hi)]
        if not clipped.empty:
            warnings.warn(
                f"yrange={yrange} clips {len(clipped)} {label} hull point(s) with "
                f"{col} outside that range (e.g. {clipped[col].iloc[0]:.4g}); "
                "pass a wider yrange or None to auto-scale.",
                stacklevel=3,
            )

    warn_if_clips(df_hull, energy_col, "primary")

    df_hull_dft = None
    if dft_mixing_energy_col is not None:
        df_dft_plot = df.dropna(subset=[x_col, dft_mixing_energy_col]).copy()
        df_hull_dft = compute_convex_hull(df_dft_plot, x_col=x_col, energy_col=dft_mixing_energy_col)
        warn_if_clips(df_hull_dft, dft_mixing_energy_col, "DFT")

    if yrange is None:
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
            text=df_hull[label_col].astype(str).map(_subscript_digits) if label_col else None,
            textposition="top center",
            textfont=dict(size=12, color=_structure_color()),
            line=dict(color=_HULL_COLOR, width=1.5, dash="dot"),
            marker=dict(size=7, color=_HULL_COLOR),
        )
    )

    if df_hull_dft is not None:
        dft_color = _structure_color(1)
        fig.add_trace(
            go.Scatter(
                x=df_hull_dft[x_col],
                y=df_hull_dft[dft_mixing_energy_col],
                mode="lines+markers" + ("+text" if label_col else ""),
                name="Convex hull (DFT)",
                text=df_hull_dft[label_col].astype(str).map(_subscript_digits) if label_col else None,
                textposition="bottom center",
                textfont=dict(size=12, color=dft_color),
                line=dict(color=dft_color, width=1.5, dash="dash"),
                marker=dict(size=8, symbol="square-open", color=dft_color),
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
    dft_mixing_energy_col: Optional[str] = None,
    yrange: Optional[Tuple[float, float]] = None,
    max_energy_above_hull: Optional[float] = None,
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
    dft_mixing_energy_col : Optional[str]
        Column holding a second, already-computed mixing energy to overlay
        as a comparison hull -- e.g. one built from Materials Project's own
        `energy_per_atom` (see `compute_mixing_energy`), alongside the
        primary `energy_col` (typically your own potential's energies). Drawn
        as a dashed hull with open-square markers in a second accent color
        (`_structure_color(1)`), its own labels, and a small legend
        distinguishing the two hulls -- omitted (no legend) when None, so
        the single-series case is pixel-identical to before this option
        existed. Off-hull points for this series are not plotted (only its
        hull), so the DFT overlay reads as a comparison line rather than a
        second full point cloud.
    yrange : Optional[Tuple[float, float]]
        Y-axis range as (min, max). Overrides `max_energy_above_hull`.
    max_energy_above_hull : Optional[float]
        None (default): auto-scale the y-axis to all data. When `yrange`
        is not given and this is a float instead, caps the y-axis at
        `hull_max + max_energy_above_hull` so a few badly-relaxed outliers
        can't squash the hull into a thin band at the bottom (see
        `_default_yrange`) -- opt into this if auto-scaling looks bad for
        your data, rather than it being the default.
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
        Rows of `df` on the lower hull for `energy_col` (see
        `compute_convex_hull`). The DFT overlay's hull (when
        `dft_mixing_energy_col` is given) is only plotted, not returned --
        call `compute_convex_hull` on that column yourself if you need it.
    """
    import warnings

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df_plot = df.dropna(subset=[x_col, energy_col]).copy()
    df_hull = compute_convex_hull(df_plot, x_col=x_col, energy_col=energy_col)

    def warn_if_clips(hull, col, label):
        if yrange is None:
            return
        lo, hi = yrange
        clipped = hull[(hull[col] < lo) | (hull[col] > hi)]
        if not clipped.empty:
            warnings.warn(
                f"yrange={yrange} clips {len(clipped)} {label} hull point(s) with "
                f"{col} outside that range (e.g. {clipped[col].iloc[0]:.4g}); "
                "pass a wider yrange or None to auto-scale.",
                stacklevel=3,
            )

    warn_if_clips(df_hull, energy_col, "primary")

    df_hull_dft = None
    if dft_mixing_energy_col is not None:
        df_dft_plot = df.dropna(subset=[x_col, dft_mixing_energy_col]).copy()
        df_hull_dft = compute_convex_hull(df_dft_plot, x_col=x_col, energy_col=dft_mixing_energy_col)
        warn_if_clips(df_hull_dft, dft_mixing_energy_col, "DFT")

    if yrange is None:
        yrange = _default_yrange(df_hull, energy_col, max_energy_above_hull)

    default_xlabel, default_ylabel = _default_axis_labels(x_col, energy_col, element=element, latex=True)
    xlabel = xlabel or default_xlabel
    ylabel = ylabel or default_ylabel

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    structure_color = _structure_color(0)
    # Landau's common-tangent styling for the hull (see plot_excess_free_energy):
    # black dotted line, lw=1.5, zorder=3; black hull-vertex dots, s=25,
    # zorder=7; labels via _place_labels_no_overlap's own defaults (fontsize
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

    dft_color = _structure_color(1)
    if df_hull_dft is not None:
        ax.plot(
            df_hull_dft[x_col], df_hull_dft[dft_mixing_energy_col],
            linestyle="dashed", color=dft_color, linewidth=1.5, zorder=4,
        )
        ax.scatter(
            df_hull_dft[x_col], df_hull_dft[dft_mixing_energy_col],
            marker="s", facecolors="none", edgecolors=dft_color, linewidths=1.3, s=36, zorder=8,
        )

    if label_col:
        label_entries = [
            {
                "x": row[x_col], "y": row[energy_col],
                "text": _subscript_digits(str(row[label_col])), "color": structure_color,
            }
            for _, row in df_hull.iterrows()
        ]
        if df_hull_dft is not None:
            label_entries += [
                {
                    "x": row[x_col], "y": row[dft_mixing_energy_col],
                    "text": _subscript_digits(str(row[label_col])), "color": dft_color,
                }
                for _, row in df_hull_dft.iterrows()
            ]
        _place_labels_no_overlap(fig, ax, label_entries)

    if df_hull_dft is not None:
        ax.legend(
            handles=[
                Line2D([0], [0], color=_HULL_COLOR, marker="o", linestyle="dotted", label="Convex hull"),
                Line2D(
                    [0], [0], color=dft_color, marker="s", markerfacecolor="none",
                    linestyle="dashed", label="Materials Project (DFT)",
                ),
            ],
            frameon=False, fontsize="small", loc="best",
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
        With `fields=None` (the default), the result already includes MP's
        own `formation_energy_per_atom` and `energy_above_hull` alongside
        the raw `energy_per_atom` -- no second API call needed. See
        `analyze_convex_hull`'s `dft_energy_col` for which of these is the
        right one to use for a DFT comparison.
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
    dft_energy_col: Optional[str] = None,
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
    dft_energy_col : Optional[str]
        Name of a per-atom energy column to overlay as a comparison hull.
        None (default) omits it -- the single-series plot is unchanged.
        Two columns are meaningful here when `df` comes from
        `fetch_structures_and_energies` (both present for free, no second
        API call -- `fields=None` there already fetches everything):
        - `"formation_energy_per_atom"` (recommended): Materials Project's
          own, already-hull-consistent formation energy -- the same value
          their website's phase diagram is built from. Verified directly
          against MP's own `is_stable`/`energy_above_hull` fields; using it
          here reproduces MP's actual hull rather than re-deriving one.
          (An earlier version of this feature re-derived a hull from
          `get_entries_in_chemsys` restricted to GGA/GGA+U entries -- that
          disagreed with MP's real numbers by ~10-20 meV/atom for some
          compounds, enough to misjudge which compositions are stable.
          Don't do that; this field is already correct.)
        - `"energy_per_atom"` (MP's *raw*, uncorrected energy): mixes
          entries computed with different DFT functionals (GGA/GGA+U/
          r2SCAN) on incompatible absolute scales, most visibly for the
          pure-element endpoints -- anchoring a mixing-energy calculation
          to those can put compounds far off any sane hull. Useful mainly
          to demonstrate that problem, not as a real comparison.
        Whichever column is passed, its mixing energy is computed the same
        way as the primary series (anchored to its own lowest-energy
        x=0/x=1 entries) into `f"{mixing_energy_col}_dft"`, then plotted as
        a second, dashed hull. (`formation_energy_per_atom` is already ~0
        at the endpoints, so this re-anchoring is a harmless no-op for it.)
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
    `df_hull` is always the primary (`energy_col`) hull -- the DFT overlay is
    plotted but not returned; call `compute_mixing_energy` +
    `compute_convex_hull` on `dft_energy_col` yourself if you need its hull
    as a dataframe.
    """
    df_mix = compute_mixing_energy(df, x_col=x_col, energy_col=energy_col, mixing_energy_col=mixing_energy_col)

    dft_mixing_energy_col = None
    if dft_energy_col is not None:
        dft_mixing_energy_col = f"{mixing_energy_col}_dft"
        df_mix = compute_mixing_energy(
            df_mix, x_col=x_col, energy_col=dft_energy_col, mixing_energy_col=dft_mixing_energy_col
        )

    if backend == "plotly":
        return plot_convex_hull(
            df_mix, x_col=x_col, energy_col=mixing_energy_col, label_col=label_col,
            dft_mixing_energy_col=dft_mixing_energy_col, **plot_kwargs
        )
    elif backend == "matplotlib":
        return plot_convex_hull_matplotlib(
            df_mix, x_col=x_col, energy_col=mixing_energy_col, label_col=label_col,
            dft_mixing_energy_col=dft_mixing_energy_col, **plot_kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'plotly' or 'matplotlib'.")
