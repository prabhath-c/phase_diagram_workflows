"""
Relax many structures with a LAMMPS potential; every result is a file, so the notebook can close.

``relax_structures`` is the one call. It relaxes each structure (cell and positions, or positions only)
and writes ``<working_directory>/<name>/relaxed.pkl``. ``load_relaxed_structures`` reads the folder back as
a table. That is the whole state: a structure with a ``relaxed.pkl`` is done and is never run again.

Use ``relax_volume=True`` (the default) for a pristine phase, to get its equilibrium cell and the
energies the defect formation energies are measured against. Use ``relax_volume=False`` for a structure with
a defect, which is relaxed at the fixed cell of the pristine host.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd
from ase import Atoms

from phase_diagram_workflows.defect_energies.jobs import follow_jobs, job_state, job_tracking, mark_submitted
from phase_diagram_workflows.utils.executor_spec import ExecutorSpec
from phase_diagram_workflows.utils.nested_batch import run_nested_batch

RESULT_FILE = "relaxed.pkl"
SUMMARY_FILE = "relaxed_structures.csv"
TABLE_FILE = "relaxed_structures.pkl"

StructuresInput = Union[Dict[str, Atoms], Sequence[Atoms], pd.DataFrame]


def _as_named_structures(
    structures: StructuresInput, atoms_col: str = "structure", name_col: Optional[str] = None
) -> Dict[str, Atoms]:
    """Name every structure: a dict is used as is, a DataFrame needs the `atoms_col` column (names from `name_col`, else the index)."""
    if isinstance(structures, pd.DataFrame):
        if atoms_col not in structures.columns:
            raise ValueError(f"The DataFrame has no {atoms_col!r} column holding the ase.Atoms (see atoms_col).")
        if name_col is not None and name_col not in structures.columns:
            raise ValueError(f"The DataFrame has no {name_col!r} column to take the names from (see name_col).")
        names = structures.index if name_col is None else structures[name_col]
        named = {str(name): atoms for name, atoms in zip(names, structures[atoms_col])}
        if len(named) != len(structures):
            raise ValueError("The structure names are not unique, but each one becomes its own folder.")
    elif isinstance(structures, dict):
        named = {str(name): atoms for name, atoms in structures.items()}
    else:
        named = {f"structure_{i}": atoms for i, atoms in enumerate(structures)}
    for name in named:
        if not name or os.sep in name or name in (".", ".."):
            raise ValueError(f"Structure name {name!r} is used as a folder name and cannot contain '{os.sep}'.")
    return named


def _relax_one(atoms: Atoms, potential_df: pd.DataFrame, relax_volume: bool, ftol: float) -> Atoms:
    """Relax positions, and the cell volume too if `relax_volume`, with atomistics' LAMMPS calculator."""
    from atomistics.calculators.lammps.libcalculator import (
        optimize_positions_and_volume_with_lammpslib,
        optimize_positions_with_lammpslib,
    )

    optimize = optimize_positions_and_volume_with_lammpslib if relax_volume else optimize_positions_with_lammpslib
    return optimize(structure=atoms, potential_dataframe=potential_df, ftol=ftol)


def _relax_and_save(item: tuple, potential_df: pd.DataFrame, relax_volume: bool, ftol: float) -> str:
    """Relax one structure and write its ``relaxed.pkl`` (the unit of work that runs inside a job)."""
    from atomistics.calculators.lammps.libcalculator import calc_static_with_lammpslib

    name, atoms, folder = item
    with job_tracking(folder):   # started.json now, error.txt if this raises
        relaxed = _relax_one(atoms, potential_df, relax_volume, ftol)
        energy = float(calc_static_with_lammpslib(structure=relaxed, potential_dataframe=potential_df)["energy"])

        temporary = os.path.join(folder, RESULT_FILE + ".tmp")
        with open(temporary, "wb") as handle:
            pickle.dump({"name": name, "structure": relaxed, "energy": energy, "relax_volume": relax_volume}, handle)
        os.replace(temporary, os.path.join(folder, RESULT_FILE))   # a half-written file never counts as a result
    return name


def relax_structures(
    structures: StructuresInput,
    potential_df: pd.DataFrame,
    working_directory: str,
    relax_volume: bool = True,
    ftol: float = 1e-4,
    executor_spec: Optional[ExecutorSpec] = None,
    structures_in_parallel: int = 10,
    threads_per_structure: int = 2,
    follow: bool = False,
    follow_interval: float = 15.0,
    atoms_col: str = "structure",
    name_col: Optional[str] = None,
) -> pd.DataFrame:
    """Relax every structure that has no result yet; each result is written to ``<working_directory>/<name>/relaxed.pkl``.

    The returned table is also saved there as ``relaxed_structures.pkl`` and ``.csv``.
    Rerunning from the same `working_directory` resumes: structures with a ``relaxed.pkl`` are skipped, whatever
    else changed. Only rerun when nothing from an earlier call is still queued or running, a job that is
    waiting leaves no trace on disk.

    Parameters
    ----------
    structures : dict, list or DataFrame
        ``{name: ase.Atoms}``; a list of Atoms (named ``structure_0``, ...); or a DataFrame whose
        `atoms_col` column holds the Atoms and whose `name_col` column (default: the index) gives the names.
        A name becomes the sub-folder. For the Materials Project table of
        ``structures.materials_project.build_structures_dataframe`` use ``atoms_col="structure_ase"``,
        ``name_col="material_id_underscore"`` (its ``structure`` column is a pymatgen dict, not Atoms).
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format (Config, Species columns).
    working_directory : str
        Where the per-structure folders go.
    relax_volume : bool, optional
        ``True`` (default): relax positions and the cell volume (atomistics'
        ``optimize_positions_and_volume_with_lammpslib``), for a pristine phase. ``False``: positions only
        at the given cell (``optimize_positions_with_lammpslib``), for structures with a defect.
    ftol : float, optional
        Force tolerance of the minimiser, in eV/Angstrom.
    executor_spec : ExecutorSpec, optional
        e.g. ``ExecutorSpec(SlurmClusterExecutor, {"pysqa_config_directory": ..., "cache_directory": ...,
        "resource_dict": {"queue": "cmmg", "run_time_limit": 3600}})``. With it, one batch job is submitted
        and this returns at once (if the executor can run on without this process); the batch relaxes
        `structures_in_parallel` structures at a time on its allocation. Do not put ``threads_per_core`` in the
        ``resource_dict``, it follows from `structures_in_parallel` and `threads_per_structure`.
        Without it the structures are relaxed one after the other in this process.
    structures_in_parallel, threads_per_structure : int, optional
        How many structures run at once in the batch, and how many CPUs each one uses.
    follow : bool, optional
        With an `executor_spec`: stay in this call and print every state change of every structure (queued,
        running, done, failed) every `follow_interval` seconds, until all are finished; raises, with the
        traceback, if one failed. Interrupting the cell only stops the waiting, the jobs go on. Default
        ``False``: submit and return (fire and forget); see ``load_job_status`` for the states later.

    Returns
    -------
    pd.DataFrame
        One row per input structure (index = name), as ``load_relaxed_structures`` gives it, with the
        relaxed ``structure`` and its ``energy``, plus a ``status`` column: ``"done"``, or the state of the
        job (``queued``, ``running``, ``failed``) for the others, whose other columns are empty.
        Call ``load_relaxed_structures`` again later to get the finished ones.
    """
    named = _as_named_structures(structures, atoms_col, name_col)
    folders = {name: os.path.join(working_directory, name) for name in named}

    already_done = {name for name in named if os.path.isfile(os.path.join(folders[name], RESULT_FILE))}
    to_run = [(name, named[name], folders[name]) for name in named if name not in already_done]
    names = list(named)
    if not to_run:
        return load_relaxed_structures(working_directory, names=names)

    for _, _, folder in to_run:
        mark_submitted(folder)
    task_args = (potential_df, relax_volume, ftol)
    if executor_spec is None:
        for item in to_run:
            _relax_and_save(item, *task_args)
        return load_relaxed_structures(working_directory, names=names)

    options = dict(executor_spec.kwargs)
    run_nested_batch(
        items=to_run,
        task_fn=_relax_and_save,
        outer_executor_cls=executor_spec.executor_class,
        task_args=task_args,
        outer_resource_dict=options.pop("resource_dict", None),
        inner_resource_dict={"cores": 1, "threads_per_core": threads_per_structure},
        inner_max_workers=min(structures_in_parallel, len(to_run)),
        cache_directory=options.pop("cache_directory", None),
        pysqa_config_directory=options.pop("pysqa_config_directory", None),
        wait=not executor_spec.detaches,   # an executor that cannot run on without this process has to be waited for
        **options,
    )
    if follow and executor_spec.detaches:
        follow_jobs(working_directory, names=[item[0] for item in to_run], interval=follow_interval)
    return load_relaxed_structures(working_directory, names=names)


def load_relaxed_structures(
    working_directory: str, save: bool = True, names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Read every ``<working_directory>/<name>/relaxed.pkl`` into a table (index = name).

    Columns: ``structure`` (relaxed Atoms), ``energy`` (eV), ``n_atoms``, ``energy_per_atom``,
    ``volume_per_atom``, the cell lengths ``a``, ``b``, ``c`` and angles ``alpha``, ``beta``, ``gamma``,
    ``relax_volume`` and ``status`` (``"done"``). Without `names`, only structures with a result are listed. With
    `names`, there is one row per name in that order, and a structure without a result yet has
    ``status="pending"`` and empty values. With `save`, the whole table is written to
    ``relaxed_structures.pkl`` (structures included, the file to load in later notebooks) and, without the
    ``structure`` column, to ``relaxed_structures.csv`` (to read at a glance), both in `working_directory`.
    """
    rows = []
    for name in sorted(os.listdir(working_directory)) if os.path.isdir(working_directory) else []:
        path = os.path.join(working_directory, name, RESULT_FILE)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            result = pickle.load(handle)
        atoms = result["structure"]
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        rows.append({
            "name": result["name"], "structure": atoms, "energy": result["energy"], "n_atoms": len(atoms),
            "energy_per_atom": result["energy"] / len(atoms), "volume_per_atom": atoms.get_volume() / len(atoms),
            "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma,
            "relax_volume": result["relax_volume"], "status": "done",
        })
    table = pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame(
        columns=["structure", "energy", "n_atoms", "energy_per_atom", "volume_per_atom",
                 "a", "b", "c", "alpha", "beta", "gamma", "relax_volume", "status"])
    table.index.name = "name"
    if names is not None:
        table = table.reindex(list(names))
        for name in table.index[table["status"].isna()]:
            state = job_state(os.path.join(working_directory, name))["state"]
            table.loc[name, "status"] = "pending" if state == "unknown" else state
        table.index.name = "name"
    if save and os.path.isdir(working_directory) and len(table):
        table.to_pickle(os.path.join(working_directory, TABLE_FILE))
        table.drop(columns="structure").to_csv(os.path.join(working_directory, SUMMARY_FILE))
    return table
