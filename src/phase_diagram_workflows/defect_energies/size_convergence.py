"""
Supercell-size convergence of one point defect's formation energy.

``converge_formation_energy`` takes the relaxed unit cell of a phase and one orbit of its sublattices, and for every supercell
size (a repeat of the unit cell) it computes

- the energy of the pristine supercell (a static calculation: the tiled relaxed cell is already in equilibrium),
- the energy of the same supercell with the defect, positions relaxed at the fixed cell,

and from the two the formation energy. A size is one job, with one folder and one ``result.pkl``, so rerunning
from the same `working_directory` resumes. ``load_size_convergence`` reads the folders back as a table, and
``plot_size_convergence`` plots it.

The defect is one representative of the orbit (``defect_from_orbit``), described in terms of the **unit cell**, so the same
description serves every size: the atom index or the void position of the unit cell is placed in the first tile of the
supercell, which starts at the origin. Before anything is submitted, the orbit is tiled onto every supercell with
``tile_atomic_sublattices`` / ``tile_interstitial_sublattices`` and the defect's site is checked to be a member of it.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from ase import Atoms

from phase_diagram_workflows.defect_energies.defect_spec import build_defect, defect_from_orbit
from phase_diagram_workflows.defect_energies.jobs import follow_jobs, job_tracking, load_job_status, mark_submitted
from phase_diagram_workflows.structures.point_defects import (
    compute_formation_energy,
    tile_atomic_sublattices,
    tile_interstitial_sublattices,
)
from phase_diagram_workflows.utils.executor_spec import ExecutorSpec

RESULT_FILE = "result.pkl"
SUMMARY_FILE = "size_convergence.csv"
DEFECT_TYPES = ("vacancy", "substitution", "interstitial")

Repeat = Union[int, Tuple[int, int, int]]


def _as_repeat(repeat: Repeat) -> Tuple[int, int, int]:
    return (int(repeat),) * 3 if isinstance(repeat, (int, np.integer)) else tuple(int(n) for n in repeat)


def _repeat_name(repeat: Tuple[int, int, int]) -> str:
    return "repeat_{}x{}x{}".format(*repeat)


def cores_for_size(n_atoms: int, atoms_per_core: int = 350, max_cores: int = 16, serial_below: int = 1500) -> int:
    """MPI ranks for a LAMMPS calculation on `n_atoms` atoms: one below `serial_below`, else about
    `atoms_per_core` atoms per rank, rounded to the nearest power of two (LAMMPS splits the cell over a 3D grid of
    ranks, and a power of two gives a compact grid where e.g. 15 would be thin slabs), at most `max_cores`."""
    if n_atoms < serial_below:
        return 1
    cores = 2 ** int(round(np.log2(max(n_atoms / atoms_per_core, 1.0))))
    return int(min(cores, 2 ** int(np.floor(np.log2(max_cores)))))


def _min_width(cell: np.ndarray) -> float:
    """Smallest perpendicular width of the cell, the distance between its opposite faces (Angstrom)."""
    volume = abs(np.linalg.det(cell))
    return float(min(volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)))


def _check_tiling(unit_cell: Atoms, orbit: Dict[str, Any], defect: Dict[str, Any], sizes: Sequence[Tuple[int, int, int]]) -> None:
    """Raise if the defect's site is not a member of `orbit` tiled onto the supercell of any of `sizes`."""
    for repeat in sizes:
        supercell = unit_cell.repeat(repeat)
        if defect["type"] == "interstitial":
            tiled = tile_interstitial_sublattices([orbit], repeat, unit_cell.get_cell()[:], supercell.get_cell()[:])[0]
            distance = np.linalg.norm(tiled["cart_positions"] - np.array(defect["position"]), axis=1).min()
            member, sites = distance < 1e-6, len(tiled["cart_positions"])
        else:
            tiled = tile_atomic_sublattices([orbit], repeat, len(unit_cell))[0]
            member, sites = defect["atom_index"] in tiled["atom_indices"], len(tiled["atom_indices"])
        if not member or sites != int(np.prod(repeat)) * orbit["multiplicity"]:
            raise ValueError(
                f"the defect {defect['label']} is not a site of orbit {orbit['label']} tiled to {_repeat_name(repeat)} "
                f"(member: {bool(member)}, {sites} sites, expected {int(np.prod(repeat)) * orbit['multiplicity']})."
            )


def _formation_energy_at_size(
    unit_cell: Atoms,
    defect: Dict[str, Any],
    repeat: Tuple[int, int, int],
    potential_df: pd.DataFrame,
    mu0: Dict[str, float],
    pinned_element: str,
    swept_element: str,
    ftol: float,
    cores: int,
    folder: str,
) -> str:
    """One size: pristine static, defect relaxation, formation energy; written to ``<folder>/result.pkl``."""
    from atomistics.calculators.lammps.libcalculator import (
        calc_static_with_lammpslib,
        optimize_positions_with_lammpslib,
    )

    with job_tracking(folder):   # started.json now, error.txt if this raises
        supercell = unit_cell.repeat(repeat)
        pristine = calc_static_with_lammpslib(structure=supercell, potential_dataframe=potential_df, cores=cores)
        defect_atoms, delta_n = build_defect(supercell, unit_cell, defect)
        relaxed = optimize_positions_with_lammpslib(
            structure=defect_atoms, potential_dataframe=potential_df, ftol=ftol, cores=cores
        )
        E_defect = float(calc_static_with_lammpslib(structure=relaxed, potential_dataframe=potential_df, cores=cores)["energy"])

        E_pristine = float(pristine["energy"])
        record = compute_formation_energy(E_defect, E_pristine, delta_n, mu0, pinned_element, swept_element)
        result = {
            "repeat": repeat,
            "n_atoms": len(supercell),
            "min_width_A": _min_width(supercell.cell.array),
            "E_pristine": E_pristine,
            "E_pristine_per_atom": E_pristine / len(supercell),
            "max_force_pristine": float(np.abs(pristine["forces"]).max()),
            "E_defect": E_defect,
            "E_formation": record["intercept"],       # at mu = mu0 for both elements (delta_mu = 0)
            "formation": record,
            "cores": cores,
            "ftol": ftol,
            "defect_structure": relaxed,
        }
        os.makedirs(folder, exist_ok=True)
        temporary = os.path.join(folder, RESULT_FILE + ".tmp")
        with open(temporary, "wb") as handle:
            pickle.dump(result, handle)
        os.replace(temporary, os.path.join(folder, RESULT_FILE))   # a half-written file never counts as a result
    return folder


def converge_formation_energy(
    unit_cell: Atoms,
    orbit: Dict[str, Any],
    kind: str,
    potential_df: pd.DataFrame,
    mu0: Dict[str, float],
    pinned_element: str,
    swept_element: str,
    repeats: Sequence[Repeat],
    working_directory: str,
    element: Optional[str] = None,
    ftol: float = 1e-8,
    atoms_per_core: int = 350,
    max_cores: int = 16,
    executor_spec: Optional[ExecutorSpec] = None,
    follow: bool = False,
    follow_interval: float = 15.0,
    rcut: Optional[float] = None,
) -> pd.DataFrame:
    """Formation energy of one defect at every supercell size that has no result yet.

    Every size is its own job, with the cores that size needs (``cores_for_size``), writing
    ``<working_directory>/repeat_<nx>x<ny>x<nz>/result.pkl``. Rerunning from the same `working_directory`
    resumes: sizes with a ``result.pkl`` are skipped. Only rerun when nothing from an earlier call is still
    queued or running, a job that is waiting leaves no trace on disk.

    Parameters
    ----------
    unit_cell : Atoms
        The relaxed unit cell of the phase; the supercells are ``unit_cell.repeat(repeat)``.
    orbit : dict
        An entry of the sublattices found on `unit_cell` (``discover_atomic_sublattices`` for a vacancy or a
        substitution, ``discover_interstitial_sublattices`` for an interstitial); its first member is the defect.
    kind : {"vacancy", "substitution", "interstitial"}
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format (Config, Species columns).
    mu0 : dict
        Energy per atom of the pure elements, ``{"Al": ..., "Mg": ...}``.
    pinned_element, swept_element : str
        As in ``compute_formation_energy``. The convergence is of the formation energy at ``mu = mu0``.
    repeats : sequence of int or (int, int, int)
        The sizes: an int ``n`` is ``(n, n, n)``.
    working_directory : str
        Where the per-size folders go.
    element : str, optional
        The element that replaces the atom (substitution) or is inserted (interstitial); see ``defect_from_orbit``.
    ftol : float, optional
        Force tolerance of the defect relaxation (eV/Angstrom). Default 1e-8: the library default of 1e-4 let
        the minimiser stop on the shoulder of an interstitial's energy landscape in the earlier size studies.
    atoms_per_core, max_cores : int, optional
        Parameters of ``cores_for_size``.
    executor_spec : ExecutorSpec, optional
        e.g. ``ExecutorSpec(SlurmClusterExecutor, {"pysqa_config_directory": ..., "cache_directory": ...,
        "resource_dict": {"queue": "cmmg", "run_time_limit": 3600}})``. With it, one job per size is submitted,
        each with its own cores, and this returns once they are submitted (if the executor can run on without
        this process). ``cores`` and ``threads_per_core`` of the ``resource_dict`` are set per size. Without it
        the sizes run one after the other in this process, on one core.
    follow : bool, optional
        With an `executor_spec`: stay in this call and print every state change of every size (queued, running,
        done, failed) every `follow_interval` seconds, until all are finished; raises, with the traceback, if one
        failed. Interrupting the cell only stops the waiting, the jobs go on. Default ``False``: submit and
        return (fire and forget); see ``load_job_status`` for the states later.
    rcut : float, optional
        Cutoff of the potential (Angstrom). Adds ``edge_clears_2rcut`` to the table, see ``load_size_convergence``.

    Returns
    -------
    pd.DataFrame
        The sizes that are done, as ``load_size_convergence`` gives it.

    Raises
    ------
    ValueError
        For a `kind`, `orbit` or `element` that do not fit, a repeated size, or a defect that is not a site of the
        tiled orbit at some size (checked before anything is run).
    """
    defect = defect_from_orbit(orbit, kind, element)    # raises for a kind, orbit or element that do not fit
    sizes = [_as_repeat(repeat) for repeat in repeats]
    if len(set(sizes)) != len(sizes):
        raise ValueError("repeats contains the same size twice.")
    _check_tiling(unit_cell, orbit, defect, sizes)

    jobs = []
    for repeat in sizes:
        folder = os.path.join(working_directory, _repeat_name(repeat))
        if not os.path.isfile(os.path.join(folder, RESULT_FILE)):
            jobs.append((repeat, folder, cores_for_size(len(unit_cell) * int(np.prod(repeat)), atoms_per_core, max_cores)))

    for _, folder, _ in jobs:
        mark_submitted(folder)
    arguments = (unit_cell, defect)
    fixed = (potential_df, mu0, pinned_element, swept_element, ftol)
    if executor_spec is None:
        for repeat, folder, _ in jobs:
            _formation_energy_at_size(*arguments, repeat, *fixed, 1, folder)
    elif jobs:
        options = dict(executor_spec.kwargs)
        base_resources = dict(options.pop("resource_dict", None) or {})
        executor = executor_spec.executor_class(resource_dict=base_resources, **options) if base_resources else executor_spec.executor_class(**options)
        stamp = int(time.time())
        for repeat, folder, cores in jobs:
            executor.submit(
                _formation_energy_at_size, *arguments, repeat, *fixed, cores, folder,
                resource_dict={
                    **base_resources, "cores": 1, "threads_per_core": cores,
                    "cache_key": f"{_repeat_name(repeat)}_{abs(hash(os.path.abspath(folder))) % 10**8}_{stamp}",   # a fresh job each submit, never a cached one
                },
            )
        executor.shutdown(wait=not executor_spec.detaches)   # one that cannot run on without this process has to be waited for
        if follow and executor_spec.detaches:
            follow_jobs(
                working_directory, names=[os.path.basename(folder) for _, folder, _ in jobs],
                interval=follow_interval,
            )
    return load_size_convergence(working_directory, rcut=rcut)


def load_size_convergence(working_directory: str, rcut: Optional[float] = None, save: bool = True) -> pd.DataFrame:
    """Read every ``<working_directory>/repeat_*/result.pkl`` into a table, smallest supercell first.

    Columns: ``repeat``, ``n_atoms``, ``min_width_A`` (smallest perpendicular width of the supercell),
    ``E_pristine``, ``E_pristine_per_atom``, ``max_force_pristine`` (should be ~0: the pristine supercell is
    already in equilibrium), ``E_defect``, ``E_formation`` (eV, at mu = mu0), ``delta_meV`` (change of
    ``E_formation`` from the previous size, once it is a few meV or less the curve has flattened), ``cores``
    and ``defect_structure`` (the relaxed supercell with the defect). With `rcut`, ``edge_clears_2rcut`` says
    whether ``min_width_A >= 2 * rcut``, below that the defect can interact with its own image directly through
    the potential. With `save`, the table without ``defect_structure`` is written to ``size_convergence.csv``.
    """
    rows = []
    for name in sorted(os.listdir(working_directory)) if os.path.isdir(working_directory) else []:
        path = os.path.join(working_directory, name, RESULT_FILE)
        if name.startswith("repeat_") and os.path.isfile(path):
            with open(path, "rb") as handle:
                result = pickle.load(handle)
            rows.append({key: result[key] for key in (
                "repeat", "n_atoms", "min_width_A", "E_pristine", "E_pristine_per_atom", "max_force_pristine",
                "E_defect", "E_formation", "cores", "defect_structure")})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("n_atoms").reset_index(drop=True)
    table["delta_meV"] = table["E_formation"].diff() * 1000
    if rcut is not None:
        table["edge_clears_2rcut"] = table["min_width_A"] >= 2 * rcut
    if save:
        table.drop(columns="defect_structure").to_csv(os.path.join(working_directory, SUMMARY_FILE), index=False)
    return table


def select_converged_size(table: pd.DataFrame, tolerance_meV: float = 1.0) -> Optional[Dict[str, Any]]:
    """The first supercell, smallest first, whose formation energy is within `tolerance_meV` of the largest one.

    The largest size is the reference and is not a candidate. If the table has ``edge_clears_2rcut``, the cell also
    has to be wide enough for the potential. None if no size qualifies, i.e. the formation energy is still more than
    the tolerance away from the largest size at every size: the sweep needs bigger cells.

    Returns
    -------
    dict or None
        ``repeat``, ``n_atoms``, ``E_formation`` (eV), ``distance_to_largest_meV``, ``tolerance_meV``.
    """
    if len(table) < 2:
        return None
    table = table.sort_values("n_atoms").reset_index(drop=True)
    distance = (table["E_formation"] - table["E_formation"].iloc[-1]).abs() * 1000
    wide = table["edge_clears_2rcut"] if "edge_clears_2rcut" in table.columns else pd.Series(True, index=table.index)
    for i in range(len(table) - 1):
        if distance[i] < tolerance_meV and wide[i]:
            row = table.iloc[i]
            return {
                "repeat": tuple(int(n) for n in row["repeat"]), "n_atoms": int(row["n_atoms"]),
                "E_formation": float(row["E_formation"]), "distance_to_largest_meV": float(distance[i]),
                "tolerance_meV": float(tolerance_meV),
            }
    return None


def plot_size_convergence(
    table: pd.DataFrame, title: Optional[str] = None, rcut: Optional[float] = None, tolerance_meV: float = 1.0
) -> Any:
    """Two panels against the supercell size, as in the earlier size studies.

    1. The formation energy, with a dashed line at its value for the largest size.
    2. The distance of the formation energy from the largest size, in meV on a log axis, with a dashed line at
       `tolerance_meV`.

    With ``edge_clears_2rcut`` in the table (see ``load_size_convergence``) and some sizes narrower and some wider than
    2 x `rcut`, a dotted vertical line marks the size at which the cell width reaches 2 x `rcut`; to its left the
    defect can see its own image directly through the potential. The line is placed at ``N_a (2 rcut / w_a)^3``, with
    `N_a` and `w_a` the atoms and width of the largest cell that is too narrow (the width grows as ``N^(1/3)`` for the
    isotropic repeats of the sweep). The pristine energy per atom, which has to be the same at every size, is a
    column of the table, not a panel. Returns the figure.
    """
    import matplotlib.pyplot as plt

    orange, red, muted, ink, grid, surface = "#eb6834", "#c0392b", "#898781", "#0b0b0b", "#e1e0d9", "#fcfcfb"
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=ink, labelsize=9)
        ax.set_xlabel("supercell size (atoms)", color=ink)
    n_atoms = table["n_atoms"].to_numpy()
    energy = table["E_formation"].to_numpy()

    axes[0].plot(n_atoms, energy, "o-", color=orange, linewidth=2, markersize=6, zorder=3)
    axes[0].axhline(energy[-1], color=muted, linewidth=1.2, linestyle="--", zorder=2)
    axes[0].set_ylabel("formation energy (eV per defect)", color=ink)
    axes[0].set_title("formation energy vs. size", color=ink, fontsize=10)

    distance = np.abs(energy[:-1] - energy[-1]) * 1000
    axes[1].plot(n_atoms[:-1], np.where(distance > 0, distance, np.nan), "o-", color=red, linewidth=2, markersize=6, zorder=3)
    axes[1].axhline(tolerance_meV, color=muted, linewidth=1.0, linestyle="--", zorder=2, label=f"{tolerance_meV:g} meV per defect")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$|E_f(N) - E_f(N_{\mathrm{largest}})|$ (meV per defect)", color=ink)
    axes[1].set_title(f"convergence vs. N={int(n_atoms[-1])} reference", color=ink, fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)

    note = f"left panel dashed line: value at N={int(n_atoms[-1])} atoms = {energy[-1]:.4f} eV."
    if "edge_clears_2rcut" in table.columns and rcut is not None:
        clears = table["edge_clears_2rcut"].to_numpy()
        if clears.any() and not clears.all():
            last_narrow = np.flatnonzero(~clears)[-1]
            at = n_atoms[last_narrow] * (2 * rcut / table["min_width_A"].to_numpy()[last_narrow]) ** 3
            for ax in axes:
                ax.axvline(at, color=orange, linewidth=1.2, linestyle=":", zorder=2, label=f"cell width = 2 x rcut ({2 * rcut:g} A)")
            axes[1].legend(frameon=False, fontsize=8)
            note += f"   dotted line: cell width = 2 x rcut ({2 * rcut:g} A); to its left the defect can see its own image directly."
    figure.text(0.5, -0.02, note, ha="center", fontsize=8, color=muted)
    if title:
        figure.suptitle(title, color=ink)
    return figure


CONVERGED_FILE = "converged_size.json"


def summarize_size_convergence(
    working_directory: str,
    name: str,
    output_directory: str,
    tolerance_meV: float = 1.0,
    rcut: Optional[float] = None,
    title: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Table, plot and converged size of a finished (or running) ``converge_formation_energy``.

    Reads the results in `working_directory` (``load_size_convergence``) and writes, into `output_directory`
    (the notebook's folder): ``<name>.png`` (``plot_size_convergence``), ``<name>.pkl`` (the table without the relaxed
    structures, which stay in `working_directory`) and ``converged_size.json`` (``select_converged_size``, with
    `extra` added, e.g. the phase, defect label and potential), which the production calculation of this defect type
    loads. If the sweep is not converged, no ``converged_size.json`` is left: one from an earlier sweep would be stale.

    Returns
    -------
    (pd.DataFrame, dict or None)
        The table, and the converged size (None if there is none).
    """
    import json

    import matplotlib.pyplot as plt

    table = load_size_convergence(working_directory, rcut=rcut)
    if table.empty:
        raise ValueError(f"no results in {working_directory} yet.")
    states = load_job_status(working_directory)["state"].value_counts().to_dict()
    print("jobs:", states)

    plot_size_convergence(table, title=title, rcut=rcut, tolerance_meV=tolerance_meV)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"{name}.png"), dpi=150, bbox_inches="tight")
    table.drop(columns="defect_structure").to_pickle(os.path.join(output_directory, f"{name}.pkl"))

    converged = select_converged_size(table, tolerance_meV)
    path = os.path.join(output_directory, CONVERGED_FILE)
    if converged is None:
        if os.path.isfile(path):
            os.remove(path)
        print(f"not converged within {tolerance_meV} meV at the largest size: extend the sizes, nothing saved")
    else:
        converged = {**converged, **(extra or {})}
        with open(path, "w") as handle:
            json.dump(converged, handle, indent=1)
        print(
            f"converged at repeat {converged['repeat']} ({converged['n_atoms']} atoms): E_f = {converged['E_formation']:.4f} eV, "
            f"{converged['distance_to_largest_meV']:.2f} meV from the largest size -> {CONVERGED_FILE}"
        )
    return table, converged
