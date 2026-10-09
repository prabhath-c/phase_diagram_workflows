"""
Point-defect diagrams of one phase: formation energies, concentrations and the temperature window of every defect, from the
records of ``defect_energies.calculate_formation_energies``.

Every defect has a formation energy that is linear in the chemical potential of one reservoir::

    E_f(dmu) = intercept + slope * dmu,        slope = -delta_n[swept element]

with the other element held at its elemental value (``intercept`` is the formation energy with both at their elemental
value). ``defect_lines`` turns the records into the table of such lines for a chosen swept element; the ``plot_*`` functions
draw the figures of the single-defect notebooks from it:

- ``plot_defect_phase_diagram``: the lines and their lower envelope, the most stable defect at each ``dmu``
- ``plot_concentration_vs_dmu``: log10 of the Boltzmann concentration against ``dmu``, one panel per temperature, and the
  dominant defect at every temperature in one figure
- ``plot_T_vs_dmu``: the temperature at which a defect reaches a concentration, against ``dmu``
- ``plot_concentration_vs_composition`` (and ``..._pdm``): the same against the bulk mole fraction ``x`` of one element

The element names come from the records, nothing here is specific to Al and Mg except the default colours.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import constants

BOLTZMANN_EV = constants.k / constants.e      # eV / K
LN10 = float(np.log(10))

# colours of a defect by (type, species); other species get FALLBACK_COLORS[type]
DEFECT_COLORS: Dict[Tuple[str, str], str] = {
    ("vacancy", "Al"): "#c0392b",          # dark red
    ("vacancy", "Mg"): "#e67e22",          # orange
    ("substitutional", "Al"): "#2980b9",   # dark blue
    ("substitutional", "Mg"): "#5dade2",   # light blue
    ("interstitial", "Al"): "#27ae60",     # green
    ("interstitial", "Mg"): "#9b59b6",     # purple
}
FALLBACK_COLORS = {"vacancy": "#c0392b", "substitutional": "#2980b9", "interstitial": "#27ae60"}
DEFECT_TYPES = ("vacancy", "substitutional", "interstitial")


# ----------------------------------------------------------------------------------------------------------------------
# the table of lines
# ----------------------------------------------------------------------------------------------------------------------


def mole_fraction(atoms, element: str) -> float:
    """Fraction of the atoms of `atoms` (an ase.Atoms) that are `element`, e.g. the composition of a phase's unit cell."""
    return atoms.get_chemical_symbols().count(element) / len(atoms)


def defect_color(defect_type: str, species: str, sublattice_int: Any = "", shade_long_interstitials: bool = False):
    """Colour of a defect. With `shade_long_interstitials`, interstitials of a site with a long label (e.g. ``18f_3``) are
    drawn paler than those of a short one (``4b``), as in the phase diagram."""
    base = DEFECT_COLORS.get((defect_type, str(species)), FALLBACK_COLORS.get(defect_type, "#888888"))
    if defect_type == "interstitial" and shade_long_interstitials:
        r, g, b = int(base[1:3], 16) / 255, int(base[3:5], 16) / 255, int(base[5:7], 16) / 255
        shade = 0.65 if len(str(sublattice_int).replace("int_", "")) > 3 else 1.0
        return (r * shade + (1 - shade), g * shade + (1 - shade), b * shade + (1 - shade))
    return base


def defect_lines(
    records: Sequence[Dict[str, Any]], swept_element: str, unique: bool = False, shade_long_interstitials: bool = False
) -> pd.DataFrame:
    """One row per defect with its line ``E_f = intercept + slope * dmu`` for the reservoir `swept_element`.

    Columns: ``defect_label``, ``defect_latex``, ``defect_type``, ``species``, ``sublattice_int``, ``delta_n_<element>`` of
    every element, ``slope`` (= -``delta_n_<swept_element>``), ``intercept`` and ``color``. The elements of the system are in
    ``table.attrs["elements"]``. With `unique`, defects with the same slope and intercept (to 4 decimals) are one row,
    the first of them.

    Raises
    ------
    ValueError
        If there are no records, or the records have no ``delta_n_<swept_element>``.
    """
    if not records:
        raise ValueError("no records")
    elements = [key[len("delta_n_"):] for key in records[0] if key.startswith("delta_n_") and key != "delta_n_total"]
    if swept_element not in elements:
        raise ValueError(f"the records have no delta_n_{swept_element}; elements are {elements}")
    rows = []
    for r in records:
        row = {k: r.get(k) for k in ("defect_label", "defect_latex", "defect_type", "species", "sublattice_int")}
        row.update({f"delta_n_{el}": r[f"delta_n_{el}"] for el in elements})
        row["slope"] = -r[f"delta_n_{swept_element}"]
        row["intercept"] = r["intercept"]
        rows.append(row)
    table = pd.DataFrame(rows)
    if unique:
        table = table.round({"slope": 4, "intercept": 4}).drop_duplicates(subset=["slope", "intercept"]).reset_index(drop=True)
    table["color"] = [
        defect_color(t, s, si, shade_long_interstitials)
        for t, s, si in zip(table["defect_type"], table["species"], table["sublattice_int"])
    ]
    table.attrs["elements"] = elements
    table.attrs["swept_element"] = swept_element
    return table


def _efs(table: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """Formation energies, shape (defects, len(x))."""
    return np.array([row["slope"] * x + row["intercept"] for _, row in table.iterrows()])


def _colormap(name: str, n: Optional[int] = None):
    """The named matplotlib colormap, resampled to `n` colours if given (``matplotlib.cm.get_cmap`` is gone from newer matplotlib)."""
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return cmap.resampled(n) if n is not None else cmap


def _legend_handles(elements: Sequence[str], extra: Sequence[Any] = ()):
    from matplotlib.lines import Line2D

    handles = []
    for kind, word in (("vacancy", "vacancy"), ("substitutional", "substitutional"), ("interstitial", "interstitial")):
        for el in elements:
            handles.append(Line2D([0], [0], color=defect_color(kind, el), lw=2, label=f"{el} {word}"))
    return handles + list(extra)


def _save(fig, path: Optional[str], dpi: int):
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ----------------------------------------------------------------------------------------------------------------------
# summary tables
# ----------------------------------------------------------------------------------------------------------------------


def stable_defect_table(table: pd.DataFrame, dmu_values: Iterable[float]) -> pd.DataFrame:
    """The defect with the lowest formation energy at each of `dmu_values`, and that energy."""
    rows = []
    for dmu in dmu_values:
        ef = table["slope"] * dmu + table["intercept"]
        best = ef.idxmin()
        rows.append({
            f"dmu_{table.attrs.get('swept_element', '')} (eV)": round(float(dmu), 2),
            "stable_defect": table.loc[best, "defect_label"],
            "E_f per defect (eV)": round(float(ef[best]), 4),
        })
    return pd.DataFrame(rows)


def concentration_table(table: pd.DataFrame, dmu_values: Iterable[float], temperatures: Iterable[float]) -> pd.DataFrame:
    """The dominant defect (lowest E_f) at each temperature and each of `dmu_values`, with its energy and concentration."""
    rows = []
    for T in temperatures:
        kBT = BOLTZMANN_EV * T
        for dmu in dmu_values:
            ef = table["slope"] * dmu + table["intercept"]
            best = ef.idxmin()
            log10c = -ef[best] / (kBT * LN10)
            rows.append({
                "T (K)": T,
                f"dmu_{table.attrs.get('swept_element', '')} (eV)": round(float(dmu), 2),
                "dominant defect": table.loc[best, "defect_label"],
                "E_f (eV)": round(float(ef[best]), 4),
                "log10(c)": round(float(log10c), 2),
                "c": f"{10 ** log10c:.2e}" if log10c > -300 else "~0",
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# 99: formation energies and their lower envelope
# ----------------------------------------------------------------------------------------------------------------------


def _merged_latex(group: pd.DataFrame) -> str:
    """Label of defects that have the same energy at the right edge: interstitials of one species become one label with
    their sites (``Al_i^{4b,8c}``), anything else keeps the label of the first."""
    first = group.iloc[0]
    if len(group) > 1 and set(group["defect_type"]) == {"interstitial"} and group["species"].nunique() == 1:
        sites = sorted(set(str(s).replace("int_", "") for s in group["sublattice_int"]))
        site_label = (",".join(sites[:2]) + ",…") if len(sites) > 2 else ",".join(sites)
        return rf"${first['species']}_i^{{{site_label}}}$"
    return first["defect_latex"]


def plot_defect_phase_diagram(
    table: pd.DataFrame, dmu_range: Tuple[float, float], phase: str, title: Optional[str] = None,
    max_labels: int = 10, save_to: Optional[str] = None, dpi: int = 300,
):
    """Formation energy of every defect against ``dmu`` of the swept element (thin dashed lines), their lower envelope
    (thick, coloured by the winning defect), the labels of the `max_labels` most stable at the right edge, and a legend.
    `table` is ``defect_lines`` with ``shade_long_interstitials=True``. Returns the figure."""
    import matplotlib.pyplot as plt

    swept = table.attrs["swept_element"]
    dmu_min, dmu_max = dmu_range
    x = np.linspace(dmu_min, dmu_max, 2000)
    y = _efs(table, x)
    envelope = np.min(y, axis=0)
    active_idx = np.argmin(y, axis=0)
    y_lo = min(envelope.min(), 0) - 0.3
    y_hi = y.max() + 1.5

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.07, 0.09, 0.50, 0.84])
    ax_nom = fig.add_axes([0.72, 0.05, 0.26, 0.92])
    for i, row in table.iterrows():
        ax.plot(x, y[i], linestyle="--", linewidth=1.0, color=row["color"], alpha=0.80)
    transitions = np.where(np.diff(active_idx) != 0)[0]
    seg_bounds = np.concatenate([[0], transitions + 1, [len(x)]])
    for i in range(len(seg_bounds) - 1):
        sl, sr = seg_bounds[i], seg_bounds[i + 1]
        c = table.iloc[active_idx[(sl + sr) // 2]]["color"]
        ax.plot(x[sl:sr], envelope[sl:sr], linewidth=3.0, color=c, zorder=4)
        ax.fill_between(x[sl:sr], envelope[sl:sr], y_lo, alpha=0.12, color=c)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="-")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="-")
    ax.set_xlim(dmu_min, dmu_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(rf"$\Delta\mu_{{{swept}}}$ (eV)", fontsize=18)
    ax.set_ylabel(r"$E_f$ per defect (eV)", fontsize=18)
    ax.set_title(title or f"{phase} — Single-Defect Formation Energies ({swept} chemical potential)", fontsize=20)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(axis="both", labelsize=18)

    # labels at the right edge: defects with the same energy there share one label
    edge = table.assign(_y=table["slope"] * dmu_max + table["intercept"])
    edge["_key"] = edge["_y"].round(4)
    items = []
    for _, group in edge.groupby("_key", sort=False):
        items.append({"y_actual": float(group["_y"].iloc[0]), "latex": _merged_latex(group), "color": group.iloc[0]["color"]})
    items = sorted(items, key=lambda d: d["y_actual"])[:max_labels]
    for item in items:
        item["y_placed"] = item["y_actual"]
    min_sep = 0.5
    for i in range(1, len(items)):
        if items[i]["y_placed"] - items[i - 1]["y_placed"] < min_sep:
            items[i]["y_placed"] = items[i - 1]["y_placed"] + min_sep
    if items and items[-1]["y_placed"] > y_hi + 0.5:
        items[-1]["y_placed"] = y_hi + 0.5
        for i in range(len(items) - 2, -1, -1):
            items[i]["y_placed"] = min(items[i]["y_placed"], items[i + 1]["y_placed"] - min_sep)
    label_x = dmu_max + 0.13
    for item in items:
        y_a, y_p, c = item["y_actual"], item["y_placed"], item["color"]
        ax.plot(dmu_max, y_a, "o", ms=3.0, color=c, clip_on=False, zorder=10)
        ax.plot([dmu_max + 0.01, label_x - 0.02], [y_a, y_p], "-", color=c, lw=0.65, alpha=0.60, clip_on=False, zorder=5)
        ax.text(label_x, y_p, item["latex"], fontsize=18, color=c, fontweight="bold", va="center", ha="left", clip_on=False)

    ax_nom.axis("off")
    elements = table.attrs["elements"]
    legend_rows: List[Tuple[Any, Any, Any, str]] = [(None, None, None, "Stable-defect envelope:")]
    for kind, word in (("vacancy", "vacancy"), ("substitutional", "substitutional"), ("interstitial", "interstitial")):
        legend_rows += [(defect_color(kind, el), "-", 3.0, f"{el} {word} dominates") for el in elements]
    legend_rows += [(None, None, None, ""), (None, None, None, "Individual lines:")]
    for kind, word in (("vacancy", "vacancy"), ("substitutional", "substitutional"), ("interstitial", "interstitial")):
        legend_rows += [(defect_color(kind, el), "--", 1.2, f"{el} {word}") for el in elements]
    y_leg = 0.88
    for col, ls, lw, label in legend_rows:
        if col is None:
            ax_nom.text(0.05, y_leg, label, fontsize=18, color="#555", va="top", transform=ax_nom.transAxes)
            y_leg -= 0.052
        else:
            ax_nom.plot([0.03, 0.20], [y_leg - 0.015, y_leg - 0.015], color=col, linestyle=ls, linewidth=lw,
                        transform=ax_nom.transAxes, clip_on=False)
            ax_nom.text(0.24, y_leg - 0.015, label, fontsize=18, va="center", transform=ax_nom.transAxes)
            y_leg -= 0.058
    box = dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", edgecolor="#cccccc", linewidth=0.8)
    ax_nom.text(0.5, 0.995, "Legend", ha="center", va="top", fontsize=20, fontweight="bold", transform=ax_nom.transAxes, bbox=box)
    _save(fig, save_to, dpi)
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# shared by the concentration figures
# ----------------------------------------------------------------------------------------------------------------------


def _dominant(table: pd.DataFrame, x: np.ndarray):
    """Formation energies, the dominant defect (lowest E_f, no sign filter) at each x, its energy, the x at which the
    dominant defect changes."""
    Ef = _efs(table, x)
    dom = np.argmin(Ef, axis=0)
    Ef_dom = Ef[dom, np.arange(len(x))]
    trans = x[np.where(np.diff(dom) != 0)[0]]
    return Ef, dom, Ef_dom, trans


def _zero_crossings(x, dom, Ef_dom, table):
    """(x0, row of the dominant defect, left_side) for every point where the dominant defect's E_f changes sign."""
    pos = Ef_dom > 0
    out = []
    for ci in np.where(np.diff(pos.astype(int)) != 0)[0]:
        den = Ef_dom[ci + 1] - Ef_dom[ci]
        if abs(den) < 1e-12:
            continue
        t = -Ef_dom[ci] / den
        out.append((x[ci] + t * (x[ci + 1] - x[ci]), table.iloc[dom[ci]], not pos[ci]))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 100: concentration against dmu
# ----------------------------------------------------------------------------------------------------------------------


def plot_concentration_vs_dmu(
    table: pd.DataFrame, dmu_range: Tuple[float, float], phase: str,
    temperatures: Sequence[float] = tuple(range(100, 1000, 100)), y_range: Tuple[float, float] = (-15, 0),
    envelope_temperatures: Sequence[float] = tuple(range(50, 1050, 50)),
    save_grid: Optional[str] = None, save_envelope: Optional[str] = None, dpi: int = 300,
):
    """log10 of the Boltzmann concentration ``exp(-E_f / kT)`` against ``dmu``: a 3x3 grid with one panel per temperature
    (dashed: every defect, bold: the most concentrated, dotted: where the dominant E_f crosses zero, i.e. the dilute limit
    breaks down) and one figure with the dominant defect at every temperature of `envelope_temperatures`. `table` should be
    ``defect_lines(..., unique=True)``. Returns ``(grid figure, envelope figure)``."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    swept = table.attrs["swept_element"]
    Y_LO, Y_HI = y_range
    dmu_min, dmu_max = dmu_range
    x = np.linspace(dmu_min, dmu_max, 2000)
    Ef_bg, dom_bg, Ef_dom_bg, trans_bg = _dominant(table, x)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)
    for panel_idx, T in enumerate(temperatures):
        ax = axes.flatten()[panel_idx]
        kBT = BOLTZMANN_EV * T
        for _x0, _dr, _ls in _zero_crossings(x, dom_bg, Ef_dom_bg, table):
            ax.axvline(_x0, color=_dr["color"], lw=1.0, ls=":", alpha=0.8, zorder=5)
            ax.text(_x0, Y_LO + 2.5, r"$E_f = 0$", rotation=90, fontsize=9, color=_dr["color"], ha="center", va="bottom",
                    zorder=6, bbox=dict(fc="white", ec="none", alpha=0.5, pad=0.5))
            _ann_x = _x0 - 0.03 if _ls else _x0 + 0.03
            _side = r"$\to$ spontaneous" if _ls else r"spontaneous $\leftarrow$"
            ax.text(_ann_x, Y_LO + 2.5, _dr["defect_latex"] + " " + _side, fontsize=8, color=_dr["color"],
                    ha="right" if _ls else "left", va="bottom", rotation=90)
        Ef_mat = _efs(table, x)
        log10c_mat = np.where(Ef_mat > 0, -Ef_mat / (kBT * LN10), np.nan)
        log10c_4mx = np.where(np.isnan(log10c_mat), -np.inf, log10c_mat)
        for i, (_, row) in enumerate(table.iterrows()):
            ax.plot(x, np.clip(log10c_mat[i], Y_LO, Y_HI), color=row["color"], lw=0.9, ls="--", alpha=0.75)
        envelope = np.max(log10c_4mx, axis=0)
        active_idx = np.argmax(log10c_4mx, axis=0)
        envelope = np.where(np.all(np.isnan(log10c_mat), axis=0), np.nan, envelope)
        transitions = np.where(np.diff(active_idx) != 0)[0]
        seg_bounds = np.concatenate([[0], transitions + 1, [len(x)]])
        for i in range(len(seg_bounds) - 1):
            sl, sr = seg_bounds[i], seg_bounds[i + 1]
            row = table.iloc[active_idx[(sl + sr) // 2]]
            ax.plot(x[sl:sr], np.clip(envelope[sl:sr], Y_LO, Y_HI), lw=2.5, color=row["color"], zorder=4)
        for _tx in trans_bg:
            ax.axvline(_tx, color="gray", lw=1.0, ls="--", alpha=0.5)
        ax.axhline(0, color="black", lw=0.8, ls="-")

        label_x = dmu_max + 0.18
        seen_y = {}
        for _, row in table.iterrows():
            y_edge = float(-(row["slope"] * dmu_max + row["intercept"]) / (kBT * LN10))
            seen_y[round(y_edge, 2)] = {"y_actual": y_edge, "y": y_edge, "latex": row["defect_latex"], "color": row["color"]}
        # only labels that land in the visible window are candidates (filtered on the true value, before staggering)
        raw_items = [d for d in seen_y.values() if Y_LO + 0.3 < d["y"] < Y_HI - 0.3]
        raw_items = sorted(raw_items, key=lambda d: d["y"], reverse=True)[:10]
        for k in range(1, len(raw_items)):
            if raw_items[k - 1]["y"] - raw_items[k]["y"] < 0.8:
                raw_items[k]["y"] = raw_items[k - 1]["y"] - 0.8
        if raw_items and raw_items[-1]["y"] < Y_LO + 0.5:
            shift = (Y_LO + 0.5) - raw_items[-1]["y"]
            for it in raw_items:
                it["y"] += shift
        for item in raw_items:
            y_stag = item["y"]
            y_anchor = np.clip(item["y_actual"], Y_LO, Y_HI)
            if not (Y_LO + 0.3 < y_stag < Y_HI - 0.3):
                continue
            ax.annotate(item["latex"], xy=(dmu_max, y_anchor), xytext=(label_x, y_stag), fontsize=9, color=item["color"],
                        va="center", ha="left", clip_on=False, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=item["color"], lw=0.7, alpha=0.8))
        ax.set_title(f"T = {T} K", fontsize=13, color="black", fontweight="bold")
        ax.set_xlim(dmu_min, dmu_max)
        ax.set_ylim(Y_LO, Y_HI)
        ax.grid(True, lw=0.3, alpha=0.35)
        ax.tick_params(axis="both", labelsize=11)
        if panel_idx % 3 == 0:
            ax.set_ylabel("Defect concentration", fontsize=14)
        if panel_idx >= 6:
            ax.set_xlabel(rf"$\Delta\mu_{{{swept}}}$ (eV)", fontsize=14)
    handles = _legend_handles(table.attrs["elements"], [
        Line2D([0], [0], color="gray", lw=1.5, ls="--", label="Individual defect"),
        Line2D([0], [0], color="gray", lw=2.5, ls="-", label="Dominant defect (envelope)"),
        Line2D([0], [0], color="gray", lw=1.0, ls=":", label=r"$E_f = 0$ (dilute limit)"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13, bbox_to_anchor=(0.5, 0.0), frameon=True)
    fig.suptitle(rf"{phase} --- $\log_{{10}}(c)$ vs $\Delta\mu_{{{swept}}}$", fontsize=18, y=1.01)
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    _save(fig, save_grid, dpi)

    # one figure: the dominant defect at every temperature
    Ef_mat = _efs(table, x)
    Ef_pos = np.where(Ef_mat > 0, Ef_mat, np.inf)
    Ef_dom = np.min(Ef_pos, axis=0)
    Ef_dom = np.where(np.isinf(Ef_dom), np.nan, Ef_dom)
    fig2, ax2 = plt.subplots(figsize=(13, 7))
    for tx in trans_bg:
        ax2.axvline(tx, color="gray", lw=1.0, ls="--", alpha=0.5)
    for _x0, _dr, _ls in _zero_crossings(x, dom_bg, Ef_dom_bg, table):
        ax2.axvline(_x0, color=_dr["color"], lw=1.2, ls=":", alpha=0.8, zorder=5)
        ax2.text(_x0, 0.5 * (Y_LO + Y_HI), r"$E_f = 0$", rotation=90, fontsize=13, color=_dr["color"], ha="center",
                 va="center", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
        _ann_x = _x0 - 0.05 if _ls else _x0 + 0.05
        _side = r"$\to$ spontaneous" if _ls else r"spontaneous $\leftarrow$"
        ax2.text(_ann_x, Y_LO + 3, _dr["defect_latex"] + " " + _side, fontsize=12, color=_dr["color"],
                 ha="right" if _ls else "left", va="bottom", rotation=90)
    T_env = list(envelope_temperatures)
    _cmap_e = _colormap("coolwarm", len(T_env))
    for k, T in enumerate(T_env):
        ax2.plot(x, -Ef_dom / (BOLTZMANN_EV * T * LN10), color=_cmap_e(k / (len(T_env) - 1)), lw=1.0, alpha=0.80)
    sm = ScalarMappable(cmap=_colormap("coolwarm"), norm=Normalize(vmin=T_env[0], vmax=T_env[-1]))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2, label="Temperature (K)", pad=0.02)
    cbar.set_label("Temperature (K)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel(rf"$\Delta\mu_{{{swept}}}$ (eV)", fontsize=18)
    ax2.set_ylabel("Defect concentration", fontsize=18)
    ax2.set_title(rf"{phase} — dominant-defect $\log_{{10}}(c)$ vs $\Delta\mu_{{{swept}}}$" +
                  "\n(one curve per temperature; dotted line = $E_f \\to 0$, dilute limit)", fontsize=18)
    ax2.set_xlim(dmu_min, dmu_max)
    ax2.set_ylim(Y_LO, Y_HI)
    ax2.grid(True, lw=0.4, alpha=0.4)
    ax2.tick_params(axis="both", labelsize=16)
    fig2.subplots_adjust(top=0.88)
    _save(fig2, save_envelope, dpi)
    return fig, fig2


# ----------------------------------------------------------------------------------------------------------------------
# 101: temperature against dmu
# ----------------------------------------------------------------------------------------------------------------------


def plot_T_vs_dmu(
    table: pd.DataFrame, dmu_range: Tuple[float, float], phase: str, T_range: Tuple[float, float] = (50, 1000),
    panel_levels: Sequence[float] = tuple(np.linspace(-15, -3, 9)), iso_levels: Sequence[float] = (-3, -6, -9, -12, -15),
    save_grid: Optional[str] = None, save_iso: Optional[str] = None, dpi: int = 300,
):
    """The temperature ``T = E_f / (k |log10 c| ln 10)`` at which a defect reaches the concentration ``c`` against ``dmu``:
    a 3x3 grid with one panel per concentration of `panel_levels` (log10 c) showing every defect and, bold with the
    accessible region, the dominant one; and one figure with the iso-concentration lines of the dominant defect for
    `iso_levels`. `table` should be ``defect_lines(..., unique=True)``. Returns ``(grid figure, iso figure)``."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    swept = table.attrs["swept_element"]
    dmu_min, dmu_max = dmu_range
    T_min, T_max = T_range
    dmu_arr = np.linspace(dmu_min, dmu_max, 800)
    Ef_1d, dom_1d, Ef_dom_bg, trans_x = _dominant(table, dmu_arr)
    Ef_pos = np.where(Ef_1d > 0, Ef_1d, np.inf)
    Ef_dom = np.min(Ef_pos, axis=0)
    Ef_dom = np.where(np.isinf(Ef_dom), np.nan, Ef_dom)
    bounds = np.concatenate([[dmu_min], trans_x, [dmu_max]])

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True, sharey=True)
    for pidx, L in enumerate(panel_levels):
        ax = axes.flatten()[pidx]
        denom = BOLTZMANN_EV * abs(L) * LN10
        for tx in trans_x:
            ax.axvline(tx, color="gray", lw=1.0, ls="--", alpha=0.5)
        for _, row in table.iterrows():
            Ef = row["slope"] * dmu_arr + row["intercept"]
            TL = np.where(Ef > 0, Ef / denom, np.nan)
            TL = np.where((TL >= T_min) & (TL <= T_max), TL, np.nan)
            ax.plot(dmu_arr, TL, color=row["color"], lw=0.9, ls="--", alpha=0.75)
        for k in range(len(bounds) - 1):
            xlo, xhi = bounds[k], bounds[k + 1]
            msk = (dmu_arr >= xlo) & (dmu_arr <= xhi)
            imid = np.argmin(np.abs(dmu_arr - 0.5 * (xlo + xhi)))
            drow = table.iloc[dom_1d[imid]]
            Ef_s = drow["slope"] * dmu_arr[msk] + drow["intercept"]
            TL_s = np.where(Ef_s > 0, Ef_s / denom, np.nan)
            ax.plot(dmu_arr[msk], np.where((TL_s >= T_min) & (TL_s <= T_max), TL_s, np.nan), color=drow["color"], lw=2.5, zorder=4)
        lbl_items = []
        for _, row in table.iterrows():
            ef_r = row["slope"] * dmu_max + row["intercept"]
            if ef_r <= 0:
                continue
            T_r = ef_r / denom
            if T_min < T_r < T_max:
                lbl_items.append({"T": T_r, "T_tip": T_r, "color": row["color"], "text": row["defect_latex"]})
        lbl_items.sort(key=lambda d: d["T"])
        lbl_items = lbl_items[:10]
        for ki in range(1, len(lbl_items)):
            if lbl_items[ki]["T"] - lbl_items[ki - 1]["T"] < 55:
                lbl_items[ki]["T"] = lbl_items[ki - 1]["T"] + 55
        for it in lbl_items:
            ax.annotate(it["text"], xy=(dmu_max, it["T_tip"]), xytext=(dmu_max + 0.08, it["T"]), fontsize=9, color=it["color"],
                        va="center", ha="left", clip_on=False, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=it["color"], lw=0.6, alpha=0.7))
        ax.set_title(f"$c = 10^{{{L:g}}}$ (iso-concentration threshold)", fontsize=13, color="black", fontweight="bold")
        ax.set_xlim(dmu_min, dmu_max)
        ax.set_ylim(T_min, T_max)
        ax.grid(True, lw=0.3, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if pidx % 3 == 0:
            ax.set_ylabel("Temperature  (K)", fontsize=14)
        if pidx >= 6:
            ax.set_xlabel(rf"$\Delta\mu_{{{swept}}}$  (eV)", fontsize=14)
    handles = _legend_handles(table.attrs["elements"], [
        Line2D([0], [0], color="gray", lw=1, ls="--", label="All defects (iso-conc line)"),
        Line2D([0], [0], color="gray", lw=2.5, ls="-", label="Dominant defect (bold)"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13, bbox_to_anchor=(0.5, 0.0), frameon=True)
    fig.suptitle(rf"{phase}  —  $T$ vs $\Delta\mu_{{{swept}}}$  (one panel per iso-concentration threshold)", fontsize=18, y=1.01)
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    _save(fig, save_grid, dpi)

    cmap_cl = _colormap("plasma_r", len(iso_levels))
    fig2, ax = plt.subplots(figsize=(11, 7))
    for k, L in enumerate(iso_levels):
        T_iso = -Ef_dom / (BOLTZMANN_EV * L * LN10)
        T_iso = np.where(T_iso <= T_max, T_iso, np.nan)
        ax.plot(dmu_arr, T_iso, color=cmap_cl(k / (len(iso_levels) - 1)), lw=1.8, label=f"$c^* = 10^{{{L}}}$")
    for _x0, _dr, _ls in _zero_crossings(dmu_arr, dom_1d, Ef_dom_bg, table):
        ax.axvline(_x0, color=_dr["color"], lw=1.2, ls=":", alpha=0.8, zorder=5)
        ax.text(_x0, 0.5 * (T_min + T_max), r"$E_f = 0$", rotation=90, fontsize=13, color=_dr["color"], ha="center",
                va="center", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
        _ann_x = _x0 - 0.05 if _ls else _x0 + 0.05
        _side = r"$\to$ spontaneous" if _ls else r"spontaneous $\leftarrow$"
        ax.text(_ann_x, T_min + 60, _dr["defect_latex"] + " " + _side, fontsize=12, color=_dr["color"],
                ha="right" if _ls else "left", va="bottom", rotation=90)
    for tx in trans_x:
        ax.axvline(tx, color="gray", lw=1.0, ls="--", alpha=0.5)
    levels_text = ", ".join(f"10^{{{L}}}" for L in iso_levels)
    ax.legend(fontsize=13, loc="upper left", framealpha=0.8)
    ax.set_xlabel(rf"$\Delta\mu_{{{swept}}}$  (eV)", fontsize=18)
    ax.set_ylabel("Temperature  (K)", fontsize=18)
    ax.set_title(rf"{phase}  —  $T$ vs $\Delta\mu_{{{swept}}}$:  iso-concentration lines" +
                 f"\n$c^* \\in \\{{{levels_text}\\}}$  (chosen reference levels)", fontsize=18)
    ax.set_xlim(dmu_min, dmu_max)
    ax.set_ylim(T_min, T_max)
    ax.grid(True, lw=0.3, alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)
    fig2.tight_layout()
    _save(fig2, save_iso, dpi)
    return fig, fig2


# ----------------------------------------------------------------------------------------------------------------------
# 102: concentration against composition
# ----------------------------------------------------------------------------------------------------------------------


def _composition_coefficients(table: pd.DataFrame, element: str, x0: float) -> np.ndarray:
    """Change of the mole fraction of `element` per defect, ``dn_el * (1 - x0) - dn_other * x0`` (binary system)."""
    others = [e for e in table.attrs["elements"] if e != element]
    if len(others) != 1:
        raise ValueError("the composition axis needs a binary system")
    dn_el = table[f"delta_n_{element}"].to_numpy(dtype=float)
    dn_other = table[f"delta_n_{others[0]}"].to_numpy(dtype=float)
    return dn_el * (1 - x0) - dn_other * x0


def plot_concentration_vs_composition(
    table: pd.DataFrame, dmu_range: Tuple[float, float], phase: str, x0: float,
    temperatures: Sequence[float] = tuple(range(100, 1000, 100)), y_range: Tuple[float, float] = (-15, 0),
    save_to: Optional[str] = None, dpi: int = 300,
):
    """log10 of the concentration of every defect against the bulk mole fraction ``x`` of the swept element, which follows
    from the defect concentrations themselves (``x = x0 + sum(c_defect * dx_defect)``) as ``dmu`` is swept; one panel per
    temperature, `x0` is the composition of the pristine phase. `table` should be ``defect_lines(..., unique=True)``.
    Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    element = table.attrs["swept_element"]
    Y_LO, Y_HI = y_range
    dmu_min, dmu_max = dmu_range
    x_coeff = _composition_coefficients(table, element, x0)
    dmu_arr = np.linspace(dmu_min, dmu_max, 2000)
    Ef_mat = _efs(table, dmu_arr)
    c900 = np.where(Ef_mat > 0, np.exp(-Ef_mat / (BOLTZMANN_EV * 900)), 0.0)
    xs900 = x0 + (x_coeff[:, None] * c900).sum(axis=0)
    xpad = max(abs(xs900 - x0).max() * 0.05, 1e-4)
    X_LO = max(0.0, xs900.min() - xpad)
    X_HI = min(1.0, xs900.max() + xpad)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)
    for panel_idx, T in enumerate(temperatures):
        ax = axes.flatten()[panel_idx]
        kBT = BOLTZMANN_EV * T
        c_for_x = np.where(Ef_mat > 0, np.exp(-Ef_mat / kBT), 0.0)
        x_bulk = x0 + (x_coeff[:, None] * c_for_x).sum(axis=0)
        log10c_mat = np.where(Ef_mat > 0, -Ef_mat / (kBT * LN10), np.nan)
        ax.axvline(x0, color="black", lw=1.2, ls="-", zorder=3)
        ax.text(x0, Y_HI - 0.5, "stoich", ha="center", va="top", fontsize=10, color="black",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
        for i, (_, row) in enumerate(table.iterrows()):
            ax.plot(x_bulk, np.clip(log10c_mat[i], Y_LO, Y_HI), color=row["color"], lw=0.9, ls="--", alpha=0.75)
        log10c_4mx = np.where(np.isnan(log10c_mat), -np.inf, log10c_mat)
        envelope = np.max(log10c_4mx, axis=0)
        active_idx = np.argmax(log10c_4mx, axis=0)
        envelope = np.where(np.all(np.isnan(log10c_mat), axis=0), np.nan, envelope)
        transitions = np.where(np.diff(active_idx) != 0)[0]
        seg_bounds = np.concatenate([[0], transitions + 1, [len(dmu_arr)]])
        for si in range(len(seg_bounds) - 1):
            sl, sr = seg_bounds[si], seg_bounds[si + 1]
            row = table.iloc[active_idx[(sl + sr) // 2]]
            ax.plot(x_bulk[sl:sr], np.clip(envelope[sl:sr], Y_LO, Y_HI), lw=2.5, color=row["color"], zorder=4)
        label_x = X_HI + (X_HI - X_LO) * 0.03
        seen_y = {}
        for _, row in table.iterrows():
            ef_r = row["slope"] * dmu_max + row["intercept"]
            if ef_r <= 0:
                continue
            y_r = -ef_r / (kBT * LN10)
            if not (Y_LO < y_r < Y_HI):
                continue
            seen_y[round(y_r, 2)] = {"y": y_r, "y_actual": y_r, "latex": row["defect_latex"], "color": row["color"]}
        items = sorted(seen_y.values(), key=lambda d: d["y"], reverse=True)[:10]
        for ki in range(1, len(items)):
            if items[ki - 1]["y"] - items[ki]["y"] < 0.8:
                items[ki]["y"] = items[ki - 1]["y"] - 0.8
        for it in items:
            ax.annotate(it["latex"], xy=(X_HI, it["y_actual"]), xytext=(label_x, it["y"]), fontsize=9, color=it["color"],
                        va="center", ha="left", clip_on=False, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=it["color"], lw=0.7, alpha=0.8))
        ax.set_title(f"T = {T} K", fontsize=13, color="black", fontweight="bold")
        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(Y_LO, Y_HI)
        ax.grid(True, lw=0.3, alpha=0.35)
        ax.tick_params(axis="both", labelsize=11)
        if panel_idx % 3 == 0:
            ax.set_ylabel("Defect concentration", fontsize=14)
        if panel_idx >= 6:
            ax.set_xlabel(rf"$x_\mathrm{{{element}}}$ (bulk {element} mole fraction)", fontsize=14)
    handles = _legend_handles(table.attrs["elements"], [
        Line2D([0], [0], color="gray", lw=1, ls="--", label="Individual defect"),
        Line2D([0], [0], color="gray", lw=2.5, label="Dominant (envelope)"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13, bbox_to_anchor=(0.5, 0.0), frameon=True)
    fig.suptitle(rf"{phase} --- log10(c) vs bulk {element} mole fraction", fontsize=18, y=1.01)
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    _save(fig, save_to, dpi)
    return fig


def plot_concentration_vs_composition_pdm(
    table: pd.DataFrame, dmu_range: Tuple[float, float], phase: str, x0: float,
    temperatures: Sequence[float] = tuple(range(100, 1000, 100)), y_range: Tuple[float, float] = (-15, 0),
    save_log: Optional[str] = None, save_linear: Optional[str] = None, dpi: int = 600,
):
    """As ``plot_concentration_vs_composition`` but with the composition ``x`` as the independent variable (the point-defect
    model): at each temperature ``dmu`` is found self-consistently by inverting ``x(dmu)``, on a range symmetric around
    `x0`. Returns ``(log10 figure, linear-concentration figure)``."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    element = table.attrs["swept_element"]
    Y_LO, Y_HI = y_range
    dmu_min, dmu_max = dmu_range
    slopes = table["slope"].to_numpy()
    intercepts = table["intercept"].to_numpy()
    x_coeff = _composition_coefficients(table, element, x0)
    dmu_arr = np.linspace(dmu_min, dmu_max, 4000)
    Ef_mat = slopes[:, None] * dmu_arr[None, :] + intercepts[:, None]
    c900 = np.exp(np.clip(-Ef_mat / (BOLTZMANN_EV * 900), -300, 300))
    xs900 = np.clip(x0 + (x_coeff[:, None] * c900).sum(axis=0), 0.0, 1.0)
    dx_lo, dx_hi = x0 - xs900.min(), xs900.max() - x0
    dx = min(dx_lo, dx_hi) * 0.97
    if dx < 1e-6:
        dx = max(dx_lo, dx_hi) * 0.47
    X_LO = max(0.0, x0 - dx * 1.03)
    X_HI = min(1.0, x0 + dx * 1.03)
    x_targets = np.linspace(X_LO, X_HI, 500)

    def stagger(items, y_lo, y_hi, gap=0.8):
        items = sorted(items, key=lambda d: d["y"])
        for k in range(1, len(items)):
            if items[k]["y"] - items[k - 1]["y"] < gap:
                items[k]["y"] = items[k - 1]["y"] + gap
        margin = (y_hi - y_lo) * 0.033     # overflow margin scaled to the axis range
        if items and items[-1]["y"] > y_hi - margin:
            sh = items[-1]["y"] - (y_hi - margin)
            for it in items:
                it["y"] -= sh
        return items

    def solved(T):
        kBT = BOLTZMANN_EV * T
        c_full = np.exp(np.clip(-Ef_mat / kBT, -300, 300))
        x_full = np.clip(x0 + (x_coeff[:, None] * c_full).sum(axis=0), 0.0, 1.0)
        valid = (x_targets >= x_full.min()) & (x_targets <= x_full.max())
        dmu_opt = np.where(valid, np.interp(x_targets, x_full, dmu_arr), np.nan)
        dmu_safe = np.where(np.isnan(dmu_opt), 0.0, dmu_opt)
        Ef_opt = slopes[:, None] * dmu_safe[None, :] + intercepts[:, None]
        return kBT, dmu_opt, Ef_opt

    handles = _legend_handles(table.attrs["elements"], [
        Line2D([0], [0], color="gray", lw=1, ls="--", label="Individual defect"),
        Line2D([0], [0], color="gray", lw=2.5, label="Dominant (envelope)"),
    ])

    def panels(kind):
        """kind 'log': log10(c); 'linear': c. Returns the figure."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)
        if kind == "log":
            lo, hi = Y_LO, Y_HI
        else:
            kBT_hi, dmu_hi, Ef_hi = solved(temperatures[-1])
            c_hi = np.where((~np.isnan(dmu_hi))[None, :] & (Ef_hi > 0), np.exp(-Ef_hi / kBT_hi), np.nan)
            lo, hi = 0.0, float(np.nanmax(c_hi)) * 1.15
        for panel_idx, T in enumerate(temperatures):
            ax = axes.flatten()[panel_idx]
            kBT, dmu_opt, Ef_opt = solved(T)
            ok = (~np.isnan(dmu_opt))[None, :] & (Ef_opt > 0)
            vals = np.where(ok, -Ef_opt / (kBT * LN10), np.nan) if kind == "log" else np.where(ok, np.exp(-Ef_opt / kBT), np.nan)
            ax.axvline(x0, color="black", lw=1.2, ls="-", zorder=3)
            ax.text(x0, hi - 0.2 if kind == "log" else hi * 0.97, "stoich", ha="center", va="top", fontsize=10,
                    color="black", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
            for i, (_, row) in enumerate(table.iterrows()):
                ax.plot(x_targets, np.clip(vals[i], lo, hi), color=row["color"], lw=0.9, ls="--", alpha=0.75)
            v4mx = np.where(np.isnan(vals), -np.inf, vals)
            envelope = np.max(v4mx, axis=0)
            active_idx = np.argmax(v4mx, axis=0)
            envelope = np.where(np.all(np.isnan(vals), axis=0), np.nan, envelope)
            transitions = np.where(np.diff(active_idx) != 0)[0]
            seg_bounds = np.concatenate([[0], transitions + 1, [len(x_targets)]])
            for si in range(len(seg_bounds) - 1):
                sl, sr = seg_bounds[si], seg_bounds[si + 1]
                row = table.iloc[active_idx[(sl + sr) // 2]]
                ax.plot(x_targets[sl:sr], np.clip(envelope[sl:sr], lo, hi), lw=2.5, color=row["color"], zorder=4)
            gap = 0.8 if kind == "log" else (hi - lo) * 0.045
            lin_margin = (hi - lo) * 0.02
            offset = 0.025
            left_items, right_items = [], []
            for i, (_, row) in enumerate(table.iterrows()):
                y_vals = vals[i]
                valid_idx = np.where(np.isfinite(y_vals))[0]
                if len(valid_idx) == 0:
                    continue
                if row["slope"] > 0:
                    if valid_idx[0] != 0:
                        continue            # the curve does not reach the left edge
                    y_edge = float(y_vals[0])
                else:
                    if valid_idx[-1] != len(y_vals) - 1:
                        continue            # nor the right edge
                    y_edge = float(y_vals[-1])
                if kind == "linear" and not (lo + lin_margin < np.clip(y_edge, lo, hi) < hi - lin_margin):
                    continue                # indistinguishable from the axis at this linear scale
                entry = {"latex": row["defect_latex"], "color": row["color"], "y_actual": y_edge, "y": y_edge}
                (left_items if row["slope"] > 0 else right_items).append(entry)
            left_items = sorted(left_items, key=lambda d: d["y_actual"], reverse=True)[:10]
            right_items = sorted(right_items, key=lambda d: d["y_actual"], reverse=True)[:10]
            for side, items, xa, xt, ha in (("L", left_items, X_LO, X_LO - offset, "right"), ("R", right_items, X_HI, X_HI + offset, "left")):
                for item in stagger(items, lo, hi, gap=gap):
                    y_stag = item["y"]
                    y_anchor = np.clip(item["y_actual"], lo, hi)
                    if kind == "log" and not (lo + 0.3 < y_stag < hi - 0.3):
                        continue
                    ax.annotate(item["latex"], xy=(xa, y_anchor), xytext=(xt, y_stag), fontsize=9, color=item["color"],
                                va="center", ha=ha, clip_on=False, annotation_clip=False,
                                arrowprops=dict(arrowstyle="-", color=item["color"], lw=0.7, alpha=0.8))
            ax.set_title(f"T = {T} K", fontsize=13, color="black", fontweight="bold")
            ax.set_xlim(X_LO, X_HI)
            ax.set_ylim(lo, hi)
            ax.grid(True, lw=0.3, alpha=0.35)
            ax.tick_params(axis="both", labelsize=11)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if panel_idx % 3 == 0:
                ax.set_ylabel("Defect concentration", fontsize=14)
            if panel_idx >= 6:
                ax.set_xlabel(rf"$x_\mathrm{{{element}}}$ (bulk {element} mole fraction)", fontsize=14)
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13, bbox_to_anchor=(0.5, 0.0), frameon=True)
        what = r"$\log_{10}(c)$" if kind == "log" else "defect concentration"
        scale = "symmetric" if kind == "log" else "linear scale"
        fig.suptitle(rf"{phase} — {what} vs bulk {element} mole fraction (PDM, {scale})", fontsize=18, y=1.01)
        fig.tight_layout(rect=[0, 0.10, 1, 1.0])
        return fig

    fig_log = panels("log")
    _save(fig_log, save_log, dpi)
    fig_lin = panels("linear")
    _save(fig_lin, save_linear, dpi)
    return fig_log, fig_lin
