"""Plots for a temperature-scaling bracket sweep of one structure.

Needs matplotlib (the ``plotting`` extra); it is imported inside the function, so importing this
module does not require it.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed
from phase_diagram_workflows.free_energies.ts_convergence.base import ts_overlap_criterion
from phase_diagram_workflows.free_energies.ts_convergence.single import (
    _bracket_working_directory,
    load_bracket_history,
)


def plot_forward_backward(
    working_directory_root: str,
    brackets: Optional[List[Tuple[float, float]]] = None,
    figsize_per_bracket: Tuple[float, float] = (4.5, 3.6),
) -> Tuple[Any, np.ndarray]:
    """Forward and backward energy difference against temperature, one panel per bracket.

    The two sweeps of a converged bracket lie on top of each other. A gap between them is
    dissipation, and the gap at the end of the sweep is what the convergence criterion measures
    (its value is in each panel title). Panels are ordered from the widest bracket to the narrowest.

    Parameters
    ----------
    working_directory_root : str
        Folder of one structure, as passed to the ``refine_temperature_bracket_*`` functions.
    brackets : Optional[List[Tuple[float, float]]], optional
        Which ``(t_low, t_high)`` brackets to draw. Default: every bracket in ``bracket_log.csv``.
    figsize_per_bracket : Tuple[float, float], optional
        Width and height of one panel.

    Returns
    -------
    Tuple[Figure, np.ndarray]
        The figure and its axes (shape ``(1, n_brackets)``). A bracket that has no output yet (still
        running, or never started) gets an empty panel saying so.
    """
    import matplotlib.pyplot as plt

    if brackets is None:
        brackets, _ = load_bracket_history(working_directory_root)
    if not brackets:
        raise ValueError(f"No brackets found in {working_directory_root}")
    brackets = sorted(brackets, key=lambda bracket: -bracket[1])

    fig, axes = plt.subplots(
        1,
        len(brackets),
        figsize=(figsize_per_bracket[0] * len(brackets), figsize_per_bracket[1]),
        squeeze=False,
    )
    for ax, (t_low, t_high) in zip(axes[0], brackets):
        ax.set_xlabel("Temperature [K]")
        try:
            result = gather_calphy_results_detailed(_bracket_working_directory(working_directory_root, t_low, t_high)).iloc[0]
        except FileNotFoundError:
            ax.set_title(f"{t_low:.0f}-{t_high:.0f} K (no output yet)")
            continue
        if result["forward_energy_diff"] is None or result["backward_energy_diff"] is None:
            ax.set_title(f"{t_low:.0f}-{t_high:.0f} K (running)")
            continue

        forward = np.asarray(result["forward_energy_diff"][0])
        backward = np.asarray(result["backward_energy_diff"][0])
        # reversible scaling: lambda = t_low / T, from 1 at t_low down to t_low / t_high
        ax.plot(t_low / np.asarray(result["forward_lambda"][0]), forward, label="forward")
        ax.plot(t_low / np.asarray(result["backward_lambda"][0]), backward, label="backward")
        ax.set_title(f"{t_low:.0f}-{t_high:.0f} K, criterion={ts_overlap_criterion(forward, backward):.4f}")
        ax.legend()
    axes[0][0].set_ylabel("Energy difference [eV/atom]")
    fig.tight_layout()
    return fig, axes
