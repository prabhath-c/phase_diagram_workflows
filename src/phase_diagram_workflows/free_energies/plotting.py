"""Plots of composition-scaling jobs.

Needs matplotlib (the ``plotting`` extra); it is imported inside the function, so importing this
module does not require it.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np

from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed


def plot_forward_backward_composition_scaling(
    working_directory: str,
    ax: Optional[Any] = None,
    smoothing: int = 100,
) -> Tuple[Any, Any]:
    """Forward and backward energy difference against lambda, for one composition_scaling job.

    The counterpart of the temperature-scaling ``plot_forward_backward``, with the same look. The energy
    difference is U(target) - U(reference) per atom along the switching; lambda is 1 at the reference composition
    and 0 at the target. The two sweeps of a reversible switching lie on top of each other: a gap between them is
    dissipation, and its value is in the title. Only the first iteration of the job is drawn.

    Parameters
    ----------
    working_directory : str
        The folder of the job, as passed to ``calc_free_energy_with_calphy``.
    ax : Optional[Any], optional
        Axes to draw into, e.g. to put several jobs side by side. Default: a new figure.
    smoothing : int, optional
        Number of steps of the running mean that takes out the thermal noise; 0 draws the raw data.

    Returns
    -------
    Tuple[Figure, Axes]
        A job that has no output yet (still running, or never started) gets an empty panel saying so.

    Raises
    ------
    ValueError
        If the folder is not a composition_scaling job.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        new_figure = True
    else:
        fig, new_figure = ax.figure, False
    ax.set_xlabel("lambda")
    ax.set_ylabel("Energy difference [eV/atom]")

    try:
        result = gather_calphy_results_detailed(working_directory).iloc[0]
    except FileNotFoundError:
        ax.set_title("(no output yet)")
        return fig, ax
    if result["calculation_mode"] != "composition_scaling":
        raise ValueError(f"{working_directory} is a '{result['calculation_mode']}' job, not composition_scaling")

    paths = [os.path.join(working_directory, f"{sweep}_1.dat") for sweep in ("forward", "backward")]
    if not all(os.path.exists(path) for path in paths) or not result["status"]:
        ax.set_title(f"{result['temperature']:.0f} K (running)")
        return fig, ax

    def mean(values):
        return np.convolve(values, np.ones(smoothing) / smoothing, mode="valid") if smoothing > 1 else values

    for path, label in zip(paths, ("forward", "backward")):
        u_reference, u_target, lam = np.loadtxt(path, comments="#", unpack=True)
        ax.plot(mean(lam), mean(u_target - u_reference), label=label)
    composition = "_".join(f"{element}_{fraction:.3f}" for element, fraction in result["composition"].items())
    ax.set_title(f"{result['temperature']:.0f} K, {composition}, dissipation={abs(result['dissipation']):.3e}")
    ax.legend()
    if new_figure:
        fig.tight_layout()
    return fig, ax
