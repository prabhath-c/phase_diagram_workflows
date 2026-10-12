"""Real LAMMPS check of calculate_formation_energies with the Liu Al-Mg EAM potential (needs iprpy-data)."""

import os
import sys

import numpy as np
import pytest
from ase.build import bulk

pytest.importorskip("atomistics")
pytest.importorskip("lammpsparser")

from phase_diagram_workflows.defect_energies import (
    calculate_formation_energies,
    converge_formation_energy,
    load_defect_results,
    relax_structures,
    save_defect_results,
)
from phase_diagram_workflows.structures.point_defects import (
    discover_atomic_sublattices,
    discover_interstitial_sublattices,
)


@pytest.fixture(scope="module")
def potential_df():
    from lammpsparser import get_potential_by_name

    df = get_potential_by_name(
        "1998--Liu-X-Y--Al-Mg--LAMMPS--ipr1", resource_path=os.path.join(sys.prefix, "share", "iprpy")
    ).to_frame().transpose()
    df["Config"] = df["Config"].apply(lambda cfg: [s if s.endswith("\n") else s + "\n" for s in cfg])
    return df


@pytest.fixture(scope="module")
def phase(potential_df, tmp_path_factory):
    """Relaxed FCC Al unit cell and the energies per atom of the pure elements."""
    path = tmp_path_factory.mktemp("pristine")
    cell = relax_structures({"Al": bulk("Al", cubic=True, a=4.3)}, potential_df, str(path / "Al")).loc["Al", "structure"]
    mg = relax_structures({"Mg": bulk("Mg")}, potential_df, str(path / "Mg"))
    mu0 = {"Al": float(relax_structures({"Al": cell}, potential_df, str(path / "Al2"), relax_volume=False).loc["Al", "energy"]) / 4,
           "Mg": float(mg.loc["Mg", "energy"]) / 2}
    return cell, mu0


@pytest.mark.parametrize("kind, element", [("vacancy", None), ("substitution", "Mg"), ("interstitial", "Mg")])
def test_the_energies_are_those_of_the_size_convergence_for_the_same_defect_and_size(potential_df, phase, tmp_path, kind, element):
    cell, mu0 = phase
    orbits = discover_interstitial_sublattices(cell) if kind == "interstitial" else discover_atomic_sublattices(cell)
    orbit = next(o for o in orbits if o["label"] in ("int_8c", "Al_4a"))
    records = calculate_formation_energies(
        cell, 3, [orbit], kind, potential_df, mu0, "Al", "Mg", str(tmp_path / "production"), elements=[element] if element else None
    )
    record = next(r for r in records if element is None or r["species"] == element)
    table = converge_formation_energy(
        cell, orbit, kind, potential_df, mu0, "Al", "Mg", repeats=[3], working_directory=str(tmp_path / "convergence"), element=element
    )
    assert record["intercept"] == pytest.approx(table.loc[0, "E_formation"], abs=1e-8)
    assert record["E_pristine"] == pytest.approx(table.loc[0, "E_pristine"], abs=1e-8)
    assert np.allclose(record["atoms_opt"].cell.array, cell.repeat(3).cell.array, atol=1e-12)    # positions only
    assert record["formation"]["landau"]["exact"] is (kind == "substitution")


def test_every_orbit_and_element_is_calculated_and_stored(potential_df, phase, tmp_path):
    cell, mu0 = phase
    records = calculate_formation_energies(
        cell, 2, discover_interstitial_sublattices(cell), "interstitial", potential_df, mu0, "Al", "Mg", str(tmp_path / "run")
    )
    assert len(records) == 2 * len(discover_interstitial_sublattices(cell))
    loaded = load_defect_results(save_defect_results(records, str(tmp_path), "interstitial")["zip"])
    assert sorted(r["defect_label"] for r in loaded) == sorted(r["defect_label"] for r in records)
    assert all(r["intercept"] > 0 for r in loaded)
