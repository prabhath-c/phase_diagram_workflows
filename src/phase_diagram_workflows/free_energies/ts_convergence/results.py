"""Collect the free energy of finished temperature-scaling runs: one pickle per phase, plus a table of what was found.

Reads only what is on disk (``bracket_log.csv`` and the calphy output of the finished bracket), so it can be called
at any time, from a notebook or a workflow, and as often as wanted. The full calphy output stays where it is.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import yaml

from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed
from phase_diagram_workflows.free_energies.ts_convergence.base import pick_best_converged_bracket
from phase_diagram_workflows.free_energies.ts_convergence.single import (
    _bracket_working_directory,
    load_bracket_history,
)

PICKLE_SUFFIX = "_free_energy.pkl"


def _read_calculation_input(bracket_directory: str) -> Dict[str, Any]:
    with open(os.path.join(bracket_directory, "input_file.yaml")) as handle:
        return yaml.safe_load(handle)["calculations"][0]


def _count_atoms(bracket_directory: str) -> Optional[int]:
    """Number of atoms in the structure calphy was given (the ``N atoms`` line of its lammps data file)."""
    path = os.path.join(bracket_directory, "input_structure.data")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        for line in handle:
            if line.split()[1:2] == ["atoms"]:
                return int(line.split()[0])
    return None


def collect_ts_results(
    phase_roots: Mapping[str, str],
    tolerance: float,
    output_directory: str,
) -> pd.DataFrame:
    """Save temperature and free energy of the finished bracket of every phase, one pickle each.

    For each phase the bracket that meets ``tolerance`` (see ``pick_best_converged_bracket``) is read with
    ``gather_calphy_results_detailed`` and written to ``<output_directory>/<phase>_free_energy.pkl`` as a DataFrame with
    the columns ``temperature`` [K], ``free_energy`` and ``free_energy_error`` [eV/atom] at full resolution.
    What belongs to it (composition, number of atoms, bracket, criterion, tolerance, potential, pressure, switching
    steps, seed and the folder of the full output) is in ``DataFrame.attrs``, which pickling keeps.
    A phase without a finished bracket is skipped, and says so in the returned table.

    Parameters
    ----------
    phase_roots : Mapping[str, str]
        Phase name -> the folder of its temperature-scaling run (the ``working_directory_root`` of the
        ``refine_temperature_bracket_*`` functions, with ``bracket_log.csv`` in it).
    tolerance : float
        A bracket is finished when its criterion is below this (``float("inf")``: the first bracket is).
    output_directory : str
        Where the pickles go; created if missing.

    Returns
    -------
    pd.DataFrame
        One row per phase: ``phase``, ``finished``, ``t_low``, ``t_high``, ``criterion``, ``pickle`` (the file written,
        or None).
    """
    os.makedirs(output_directory, exist_ok=True)
    rows = []
    for phase, root in phase_roots.items():
        _, criteria = load_bracket_history(str(root))
        finished = pick_best_converged_bracket(criteria, tolerance)
        if finished is None:
            rows.append({"phase": phase, "finished": False, "t_low": None, "t_high": None, "criterion": None, "pickle": None})
            continue

        t_low, t_high = finished
        bracket_directory = _bracket_working_directory(str(root), t_low, t_high)
        result = gather_calphy_results_detailed(bracket_directory).iloc[0]
        calculation = _read_calculation_input(bracket_directory)

        free_energy = pd.DataFrame({
            "temperature": result["temperature"],
            "free_energy": result["free_energy"],
            "free_energy_error": result["free_energy_error"],
        })
        free_energy.attrs = {
            "phase": phase,
            "composition": {str(element): float(fraction) for element, fraction in result["composition"].items()},
            "n_atoms": _count_atoms(bracket_directory),
            "t_low": t_low,
            "t_high": t_high,
            "criterion": float(criteria[finished]),
            "tolerance": tolerance,
            "pressure": result["pressure"],
            "potential": calculation["pair_coeff"],
            "n_switching_steps": calculation["n_switching_steps"],
            "seed": calculation["md"].get("seed"),
            "results_directory": bracket_directory,
        }
        pickle_path = os.path.join(output_directory, phase + PICKLE_SUFFIX)
        free_energy.to_pickle(pickle_path)
        rows.append({
            "phase": phase, "finished": True, "t_low": t_low, "t_high": t_high,
            "criterion": float(criteria[finished]), "pickle": pickle_path,
        })
    return pd.DataFrame(rows)
