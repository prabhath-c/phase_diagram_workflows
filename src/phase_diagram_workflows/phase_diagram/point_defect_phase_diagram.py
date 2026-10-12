"""
Phase diagram with point defects: the free energies of the temperature-scaling runs plus the formation energies of every single
point defect, combined with landau (exact site-exclusion statistics, one sublattice per symmetry-distinct site, no defect entropy).

Two kinds of phase are built, each by its own function from files the earlier steps wrote:

- ``build_defected_phase``: a stoichiometric phase. Its host is a landau ``TemperatureDependentLinePhase`` (free energy against
  temperature at one composition, ``load_line_phase``) with its point defects on top (``PointDefectedPhase``). Needs the
  ``<phase>_free_energy.pkl`` of ``free_energies.ts_convergence.collect_ts_results`` and the ``defect_energies`` results of the phase.
- ``build_interpolated_phase``: a phase known at several compositions (the liquid), one such pickle per composition. A line phase
  per pickle, interpolated in composition (``SlowInterpolatingPhase``). No defects.

``compute_phase_diagram`` runs landau on the list of phases (on SLURM if an ``ExecutorSpec`` is given) and caches the table on disk.

Layout the defect side expects: a phase's defects folder holds ``Vacancy/vacancy_results.zip``, ``Substitutional/substitutional_results.zip``
and ``Interstitial/interstitial_results.zip`` (what ``defect_energies.save_defect_results`` writes); its name is free. A host pickle
can have any name; it needs the columns ``temperature`` and ``free_energy`` and ``DataFrame.attrs["composition"]``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from phase_diagram_workflows.defect_energies.formation_energies import load_phase_defect_results
from phase_diagram_workflows.utils.executor_spec import ExecutorSpec

DEFECT_KINDS = (("Vacancy", "vacancy"), ("Substitutional", "substitutional"), ("Interstitial", "interstitial"))

# what a defect record must have (plus ``delta_n_<solute>``, see ``check_records``)
REQUIRED_RECORD_KEYS = ("defect_label", "defect_type", "sublattice_ref", "sublattice_int", "multiplicity", "n_atoms", "intercept_landau")


# ----------------------------------------------------------------------------------------------------------------------
# the host: one line phase per pickle
# ----------------------------------------------------------------------------------------------------------------------


def load_line_phase(pickle_path: str, name: str, solute: str = "Mg", n_points: int = 200):
    """The landau ``TemperatureDependentLinePhase`` of one free-energy pickle: free energy per atom against temperature at the
    pickle's composition.

    The raw free energy has one point per ~0.005 K; `n_points` of them, evenly spaced and including both ends, are kept.
    The concentration is the fraction of `solute` in ``DataFrame.attrs["composition"]``.

    Raises
    ------
    FileNotFoundError
        If `pickle_path` does not exist.
    ValueError
        If the pickle has no composition (it is not from ``collect_ts_results``) or none for `solute`.
    """
    from landau.phases import TemperatureDependentLinePhase

    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(f"free energy pickle not found: {pickle_path}")
    table = pd.read_pickle(pickle_path)
    composition = table.attrs.get("composition")
    if not composition or solute not in composition:
        raise ValueError(f"{pickle_path} has no composition for {solute!r} in its attrs; got {composition!r}.")
    keep = np.unique(np.linspace(0, len(table) - 1, n_points).round().astype(int))
    kept = table.iloc[keep]
    return TemperatureDependentLinePhase(
        name=name,
        fixed_concentration=float(composition[solute]),
        temperatures=kept["temperature"].to_numpy(),
        free_energies=kept["free_energy"].to_numpy(),
    )


def temperature_range(pickles: Union[str, Sequence[str]]) -> Tuple[float, float]:
    """(lowest, highest) temperature of a free-energy pickle; for several, the temperatures all of them cover.

    landau extrapolates a free energy beyond this, so choose the temperatures of the diagram knowing it.
    """
    pickles = [pickles] if isinstance(pickles, (str, os.PathLike)) else list(pickles)
    temperature = [pd.read_pickle(path)["temperature"] for path in pickles]
    return float(max(t.min() for t in temperature)), float(min(t.max() for t in temperature))


# ----------------------------------------------------------------------------------------------------------------------
# a phase with point defects
# ----------------------------------------------------------------------------------------------------------------------


def check_records(records: Sequence[Mapping[str, Any]], solute: str = "Mg") -> None:
    """Raise one ``ValueError`` listing every record that lacks a key ``sublattices_from_records`` needs
    (``REQUIRED_RECORD_KEYS`` and ``delta_n_<solute>``)."""
    required = (*REQUIRED_RECORD_KEYS, f"delta_n_{solute}")
    problems = [f"{r.get('defect_label', '?')}: missing {missing}" for r in records if (missing := [k for k in required if k not in r])]
    if problems:
        raise ValueError("defect records are missing keys:\n  " + "\n  ".join(problems))


def sublattices_from_records(records: Sequence[Mapping[str, Any]], solute: str = "Mg", tolerance: float = 1e-8):
    """One landau ``PointDefectSublattice`` per symmetry-distinct site, from the records of one phase.

    - the site of a record is its ``sublattice_ref`` (vacancies, substitutionals) or ``sublattice_int`` (interstitials), so the
      vacancy and the substitutionals of a site compete for it
    - the ``sublattice_fraction`` of a site is ``multiplicity / n_atoms`` of the supercell its records were computed in; the
      records of a site have supercells of different size but must agree on it
    - a defect has the energy ``intercept_landau``, no entropy and ``delta_n_<solute>`` extra solute atoms

    Raises
    ------
    ValueError
        If a record lacks a key (see ``check_records``), or the records of one site disagree on the site fraction by more
        than `tolerance`.
    """
    from landau.phases.pointdefects import ConstantPointDefect, PointDefectSublattice

    check_records(records, solute)
    by_site = defaultdict(list)
    for record in records:
        site = record["sublattice_int"] if record["defect_type"] == "interstitial" else record["sublattice_ref"]
        by_site[site].append(record)

    sublattices = []
    for index, (site, group) in enumerate(sorted(by_site.items())):
        fractions = np.array([r["multiplicity"] / r["n_atoms"] for r in group])
        if not np.allclose(fractions, fractions[0], atol=tolerance, rtol=0):
            raise ValueError(f"site {site}: the records disagree on multiplicity / n_atoms: {sorted(set(fractions.round(8)))}.")
        defects = [
            ConstantPointDefect(name=r["defect_label"], excess_energy=r["intercept_landau"], excess_entropy=0.0, excess_solutes=r[f"delta_n_{solute}"])
            for r in group
        ]
        sublattices.append(PointDefectSublattice(name=site, sublattice=index, sublattice_fraction=float(fractions[0]), defects=defects))
    return sublattices


def add_point_defects(host, defects_folder: str, name: str, solute: str = "Mg"):
    """The landau ``PointDefectedPhase`` of the line phase `host` with the point defects of the records in `defects_folder`
    (``sublattices_from_records``). The host can come from anywhere (``load_line_phase`` of a free-energy pickle, or a
    ``TemperatureDependentLinePhase`` built from another source's curve); only the defects are read here.

    `defects_folder` holds ``Vacancy/vacancy_results.zip``, ``Substitutional/substitutional_results.zip`` and
    ``Interstitial/interstitial_results.zip``; the phase is named `name` in the diagram.

    Raises
    ------
    FileNotFoundError
        If the defect results do not exist.
    """
    from landau.phases.pointdefects import PointDefectedPhase

    records = load_phase_defect_results(defects_folder)
    return PointDefectedPhase(name=name, line_phase=host, sublattices=sublattices_from_records(records, solute))


def build_defected_phase(host_pickle: str, defects_folder: str, name: str, solute: str = "Mg", n_points: int = 200):
    """A stoichiometric phase with its point defects: ``add_point_defects`` on the line phase of `host_pickle`
    (``load_line_phase``).

    Raises
    ------
    FileNotFoundError
        If the host pickle or the defect results do not exist.
    """
    host = load_line_phase(host_pickle, name, solute, n_points)
    return add_point_defects(host, defects_folder, name, solute)


# ----------------------------------------------------------------------------------------------------------------------
# a phase known at several compositions
# ----------------------------------------------------------------------------------------------------------------------


def build_interpolated_phase(pickles: Sequence[str], name: str, solute: str = "Mg", n_points: int = 200):
    """A phase known at several compositions (the liquid): one line phase per pickle (``load_line_phase``, each pickle is one
    composition), interpolated in composition by landau's ``SlowInterpolatingPhase``. It has no point defects.

    Raises
    ------
    ValueError
        If there are no pickles, or two of them have the same composition.
    FileNotFoundError
        If a pickle does not exist.
    """
    from landau.phases import SlowInterpolatingPhase

    if not len(pickles):
        raise ValueError(f"{name}: no pickles.")
    lines = [load_line_phase(path, f"{name}_x{pd.read_pickle(path).attrs['composition'][solute]:.4f}", solute, n_points) for path in pickles]
    concentrations = [line.fixed_concentration for line in lines]
    if len(set(concentrations)) != len(concentrations):
        raise ValueError(f"{name}: two pickles have the same composition: {sorted(concentrations)}.")
    return SlowInterpolatingPhase(name=name, phases=sorted(lines, key=lambda line: line.fixed_concentration))


# ----------------------------------------------------------------------------------------------------------------------
# the diagram
# ----------------------------------------------------------------------------------------------------------------------


SUBMITTED_SUFFIX = ".submitted"


def save_phase_diagram(phases: Sequence[Any], temperatures: Sequence[float], cache_path: str, n_mu: int = 200, **kwargs: Any) -> str:
    """``landau.calculate.calc_phase_diagram`` of `phases`, saved to `cache_path` (written whole, then moved into place, so a
    pickle that exists is complete). Runs where it is called: this is what ``compute_phase_diagram`` submits to SLURM.

    Extra keyword arguments go to ``calc_phase_diagram`` (default ``keep_unstable=True``). Returns `cache_path`.
    """
    from landau.calculate import calc_phase_diagram

    kwargs.setdefault("keep_unstable", True)
    table = calc_phase_diagram(list(phases), temperatures, n_mu, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    unfinished = cache_path + ".unfinished"
    table.to_pickle(unfinished)
    os.replace(unfinished, cache_path)
    return cache_path


def compute_phase_diagram(
    phases: Sequence[Any],
    temperatures: Sequence[float],
    cache_path: str,
    n_mu: int = 200,
    executor_spec: Optional[ExecutorSpec] = None,
    **kwargs: Any,
) -> Optional[pd.DataFrame]:
    """The phase diagram table of `phases`, never computed twice. Decided by files, in this order:

    1. `cache_path` exists: it is read and returned.
    2. ``<cache_path>.submitted`` exists: a job was submitted earlier and has not saved its result yet; nothing is done, None is
       returned. Delete the ``.submitted`` file to submit again (e.g. if the job failed).
    3. otherwise, with an `executor_spec`: the marker is written and ``save_phase_diagram`` is submitted to its executor, which
       is not waited for (fire and forget; the job saves `cache_path` itself). None is returned.
    4. otherwise, without an `executor_spec`: it is computed here, saved and returned.

    Extra keyword arguments go to ``save_phase_diagram``. Delete `cache_path` (and the marker) to recompute.
    """
    marker = cache_path + SUBMITTED_SUFFIX
    if os.path.isfile(cache_path):
        return pd.read_pickle(cache_path)
    if os.path.isfile(marker):
        return None
    if executor_spec is None:
        save_phase_diagram(phases, temperatures, cache_path, n_mu, **kwargs)
        return pd.read_pickle(cache_path)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(marker, "w") as handle:
        handle.write("submitted\n")
    executor = executor_spec.create()
    try:
        executor.submit(save_phase_diagram, list(phases), temperatures, cache_path, n_mu, **kwargs)
    except BaseException:
        os.remove(marker)
        raise
    finally:
        executor.shutdown(wait=False)   # the job carries on without this process
    return None
