"""
Formation energies of every point defect of one type of a phase, at one supercell size.

``calculate_formation_energies`` builds one defect per symmetry-unique site (orbit) on the supercell that the size
convergence found, relaxes all of them, positions only at the fixed cell of the pristine host, in one batch (SLURM or
in the notebook), and returns one record per defect with its formation energy as a function of the chemical
potential. ``save_defect_results`` stores the records, ``load_defect_results`` reads them back.

A record keeps the keys of the earlier single-defect pickles (``defect_label``, ``defect_latex``, ``defect_type``,
``sublattice_ref``/``sublattice_int``, ``species``, ``multiplicity``, ``unitcell_multiplicity``, ``delta_n_<element>``,
``delta_n_total``, ``E_defect``, ``slope``, ``intercept``, ``slope_mu_<pinned element>``, ``intercept_landau``,
``dmu_landau_exact``, ``mu0_<element>``, ``atoms_opt``) so the notebooks that read them keep working, and adds
``E_pristine``, ``repeat``, ``n_atoms`` and ``formation``: the full record of ``compute_formation_energy``, whose
``landau`` entry (``intercept``, ``exact``) is what ``landau.phases.pointdefects`` needs.
"""

from __future__ import annotations

import io
import os
import pickle
import zipfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from ase import Atoms

from phase_diagram_workflows.defect_energies.defect_spec import KINDS, build_defect, defect_from_orbit
from phase_diagram_workflows.defect_energies.relaxation import relax_structures
from phase_diagram_workflows.defect_energies.size_convergence import _as_repeat
from phase_diagram_workflows.structures.point_defects import compute_formation_energy
from phase_diagram_workflows.utils.executor_spec import ExecutorSpec

# the name the earlier results use for the type of a defect
DEFECT_TYPE_NAMES = {"vacancy": "vacancy", "substitution": "substitutional", "interstitial": "interstitial"}
PRISTINE = "pristine"


def _latex(kind: str, orbit: Dict[str, Any], element: Optional[str]) -> str:
    """``$V_{Al}^{4a}$``, ``$Mg_{Al}^{4a}$`` or ``$Mg_i^{8c}$``."""
    if kind == "interstitial":
        return rf"${element}_i^{{{orbit['label'][len('int_'):]}}}$"
    site = orbit["label"].split("_", 1)[1]
    if kind == "vacancy":
        return rf"$V_{{{orbit['species']}}}^{{{site}}}$"
    return rf"${element}_{{{orbit['species']}}}^{{{site}}}$"


def calculate_formation_energies(
    unit_cell: Atoms,
    repeat: Any,
    orbits: Sequence[Dict[str, Any]],
    kind: str,
    potential_df: pd.DataFrame,
    mu0: Dict[str, float],
    pinned_element: str,
    swept_element: str,
    working_directory: str,
    elements: Optional[Sequence[str]] = None,
    ftol: float = 1e-8,
    executor_spec: Optional[ExecutorSpec] = None,
    structures_in_parallel: int = 10,
    threads_per_structure: int = 2,
    follow: bool = False,
    follow_interval: float = 15.0,
) -> List[Dict[str, Any]]:
    """Formation-energy records of one defect per orbit (and element) on the supercell ``unit_cell.repeat(repeat)``.

    The pristine supercell goes through the same batch, so its energy is the one the defect energies are measured
    against. Results are written per structure to `working_directory` (see ``relax_structures``): rerunning resumes.

    Parameters
    ----------
    unit_cell : Atoms
        The relaxed unit cell of the phase.
    repeat : int or (int, int, int)
        The supercell: the converged size of the size convergence.
    orbits : sequence of dict
        The orbits found on `unit_cell`: atomic ones (``discover_atomic_sublattices``) for a vacancy or a substitution,
        interstitial ones (``discover_interstitial_sublattices``) for an interstitial.
    kind : {"vacancy", "substitution", "interstitial"}
    potential_df : pd.DataFrame
        Potential in pyiron/lammpsparser-compatible format.
    mu0 : dict
        Energy per atom of each pure element, ``{"Al": ..., "Mg": ...}``; its keys are the elements of the system.
    pinned_element, swept_element : str
        As in ``compute_formation_energy``.
    working_directory : str
        Where the per-structure folders go.
    elements : sequence of str, optional
        The elements to substitute with or to insert. Default: for a substitution every element of `mu0` except the
        orbit's own, for an interstitial every element of `mu0`. Not used for a vacancy.
    ftol : float, optional
        Force tolerance of the relaxation (eV/Angstrom). Default 1e-8: the library default of 1e-4 let the minimiser stop
        early for interstitials in the size studies.
    executor_spec, structures_in_parallel, threads_per_structure, follow, follow_interval
        As in ``relax_structures``.

    Returns
    -------
    list of dict
        One record per defect that is finished (all of them, unless the jobs were only submitted and are still running),
        in the order of `orbits`; empty while the pristine supercell is not finished. See the module docstring.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}.")
    repeat = _as_repeat(repeat)
    supercell = unit_cell.repeat(repeat)
    n_cells = int(np.prod(repeat))

    defects = []
    for orbit in orbits:
        if kind == "vacancy":
            choices: List[Optional[str]] = [None]
        elif kind == "substitution":
            choices = [e for e in (elements or list(mu0)) if e != orbit["species"]]
        else:
            choices = list(elements or list(mu0))
        for element in choices:
            spec = defect_from_orbit(orbit, kind, element)
            atoms, delta_n = build_defect(supercell, unit_cell, spec)
            defects.append((spec, orbit, element, atoms, delta_n))
    labels = [spec["label"] for spec, *_ in defects]
    if len(set(labels)) != len(labels) or PRISTINE in labels:
        raise ValueError(f"the defect labels are not unique: {labels}")

    structures = {PRISTINE: supercell, **{spec["label"]: atoms for spec, _, _, atoms, _ in defects}}
    table = relax_structures(
        structures, potential_df, working_directory,
        relax_volume=False, ftol=ftol, executor_spec=executor_spec,
        structures_in_parallel=min(structures_in_parallel, len(structures)), threads_per_structure=threads_per_structure,
        follow=follow, follow_interval=follow_interval,
    )
    if table.loc[PRISTINE, "status"] != "done":
        return []
    E_pristine = float(table.loc[PRISTINE, "energy"])

    records = []
    for spec, orbit, element, _, delta_n in defects:
        row = table.loc[spec["label"]]
        if row["status"] != "done":
            continue
        E_defect = float(row["energy"])
        formation = compute_formation_energy(E_defect, E_pristine, delta_n, mu0, pinned_element, swept_element)
        record: Dict[str, Any] = {
            "defect_label": spec["label"],
            "defect_latex": _latex(kind, orbit, element),
            "defect_type": DEFECT_TYPE_NAMES[kind],
            "sublattice_ref": orbit["label"] if kind != "interstitial" else None,
            "sublattice_int": orbit["label"] if kind == "interstitial" else None,
            "species": element if kind != "vacancy" else orbit["species"],
            "multiplicity": n_cells * orbit["multiplicity"],
            "unitcell_multiplicity": orbit["multiplicity"],
        }
        record.update({f"delta_n_{el}": delta_n.get(el, 0) for el in mu0})
        record.update({
            "delta_n_total": formation["delta_n_total"],
            "E_defect": E_defect,
            "slope": formation["slope"],
            "intercept": formation["intercept"],
            f"slope_mu_{pinned_element}": formation["slope_pinned"],
            "intercept_landau": formation["landau"]["intercept"],
            "dmu_landau_exact": formation["landau"]["exact"],
        })
        record.update({f"mu0_{el}": mu0[el] for el in mu0})
        record.update({
            "atoms_opt": row["structure"],
            "E_pristine": E_pristine,
            "repeat": repeat,
            "n_atoms": len(supercell),
            "formation": formation,
        })
        records.append(record)
    return records


def save_defect_results(records: Sequence[Dict[str, Any]], output_directory: str, name: str) -> Dict[str, str]:
    """Write ``<name>_results.zip`` (one ``<defect_label>.pkl`` per record, as the earlier results) and
    ``<name>_results.csv`` (the numbers at a glance, without the structures) into `output_directory`.

    An existing zip or csv of that name is overwritten. Returns their paths.
    """
    if not records:
        raise ValueError("no records to save.")
    zip_path = os.path.join(output_directory, f"{name}_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            archive.writestr(f"{record['defect_label']}.pkl", pickle.dumps(record))
    csv_path = os.path.join(output_directory, f"{name}_results.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("atoms_opt", "formation")} for r in records]).to_csv(csv_path, index=False)
    return {"zip": zip_path, "csv": csv_path}


def load_defect_results(zip_path: str) -> List[Dict[str, Any]]:
    """The records of a ``<name>_results.zip``, in the order they were saved."""
    with zipfile.ZipFile(zip_path) as archive:
        return [pickle.load(io.BytesIO(archive.read(member))) for member in archive.namelist()]


def load_phase_defect_results(root: str = ".") -> List[Dict[str, Any]]:
    """Every record of a phase: ``<root>/Vacancy/vacancy_results.zip``, ``<root>/Substitutional/substitutional_results.zip`` and
    ``<root>/Interstitial/interstitial_results.zip``, in that order (``root`` is the phase folder).

    Raises
    ------
    FileNotFoundError
        If none of the three archives exists, i.e. the production calculations have not been run in `root`.
    """
    records: List[Dict[str, Any]] = []
    for folder, name in (("Vacancy", "vacancy"), ("Substitutional", "substitutional"), ("Interstitial", "interstitial")):
        path = os.path.join(root, folder, f"{name}_results.zip")
        if os.path.isfile(path):
            records.extend(load_defect_results(path))
    if not records:
        raise FileNotFoundError(f"no <type>_results.zip below {os.path.abspath(root)}: run the all_* notebooks of this phase first.")
    return records
