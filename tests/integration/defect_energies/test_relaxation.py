"""Real LAMMPS check of relax_structures with the Liu Al-Mg EAM potential (needs iprpy-data)."""

import os
import sys

import pytest
from ase.build import bulk

pytest.importorskip("atomistics")
pytest.importorskip("lammpsparser")

from phase_diagram_workflows.defect_energies import load_relaxed_structures, relax_structures


@pytest.fixture(scope="module")
def potential_df():
    from lammpsparser import get_potential_by_name

    df = get_potential_by_name(
        "1998--Liu-X-Y--Al-Mg--LAMMPS--ipr1", resource_path=os.path.join(sys.prefix, "share", "iprpy")
    ).to_frame().transpose()
    df["Config"] = df["Config"].apply(lambda cfg: [s if s.endswith("\n") else s + "\n" for s in cfg])
    return df


def test_volume_relaxation_and_positions_only(potential_df, tmp_path):
    strained = bulk("Al", cubic=True, a=4.3)
    table = relax_structures({"volume": strained}, potential_df, str(tmp_path))   # default relax_volume=True
    assert table.loc["volume", "a"] == pytest.approx(4.05, abs=0.02)   # equilibrium Al lattice parameter

    fixed = relax_structures({"fixed": strained}, potential_df, str(tmp_path), relax_volume=False)
    assert fixed.loc["fixed", "a"] == pytest.approx(4.3)               # positions only: cell untouched


def test_resumes_without_rerunning_and_saves_table(potential_df, tmp_path):
    mg = bulk("Mg", a=3.2, c=5.2)
    table = relax_structures({"mg": mg}, potential_df, str(tmp_path))
    assert (tmp_path / "relaxed_structures.pkl").is_file()
    again = relax_structures({"mg": mg}, potential_df, str(tmp_path))      # nothing left to run
    assert again.loc["mg", "energy"] == table.loc["mg", "energy"]
    assert len(load_relaxed_structures(str(tmp_path))) == 1


def _al_potential_and_mu0(potential_df):
    from atomistics.calculators.lammps.libcalculator import calc_static_with_lammpslib

    unit_cell = bulk("Al", cubic=True, a=4.05)
    energy_per_atom = calc_static_with_lammpslib(structure=unit_cell, potential_dataframe=potential_df)["energy"] / 4
    return unit_cell, {"Al": float(energy_per_atom), "Mg": -1.5}


def test_size_convergence_vacancy_is_positions_only_and_matches_a_manual_calculation(potential_df, tmp_path):
    import numpy as np
    from atomistics.calculators.lammps.libcalculator import calc_static_with_lammpslib, optimize_positions_with_lammpslib

    from phase_diagram_workflows.defect_energies import converge_formation_energy
    from phase_diagram_workflows.structures.point_defects import discover_atomic_sublattices

    unit_cell = relax_structures({"Al": bulk("Al", cubic=True, a=4.3)}, potential_df, str(tmp_path / "pristine")).loc["Al", "structure"]
    mu0 = {"Al": float(calc_static_with_lammpslib(structure=unit_cell, potential_dataframe=potential_df)["energy"]) / 4, "Mg": -1.5}
    table = converge_formation_energy(
        unit_cell, discover_atomic_sublattices(unit_cell)[0], "vacancy", potential_df, mu0, "Al", "Mg",
        repeats=[2, 3], working_directory=str(tmp_path / "convergence"),
    )
    assert list(table["n_atoms"]) == [32, 108]
    # the pristine supercell is the tiled relaxed cell: same energy per atom as the unit cell, no forces
    assert np.allclose(table["E_pristine_per_atom"], mu0["Al"], atol=1e-8)
    assert (table["max_force_pristine"] < 1e-6).all()
    # positions only: the cell of the relaxed defect structure is the pristine supercell's, untouched
    for repeat, relaxed in zip((2, 3), table["defect_structure"]):
        assert np.allclose(relaxed.cell.array, unit_cell.repeat(repeat).cell.array, atol=1e-12)
    # and the formation energy is what a by-hand calculation gives
    supercell = unit_cell.repeat(2)
    defect = supercell.copy()
    del defect[0]
    relaxed = optimize_positions_with_lammpslib(structure=defect, potential_dataframe=potential_df, ftol=1e-8)
    E_defect = calc_static_with_lammpslib(structure=relaxed, potential_dataframe=potential_df)["energy"]
    by_hand = E_defect - 32 * mu0["Al"] + mu0["Al"]
    assert table.loc[0, "E_formation"] == pytest.approx(by_hand, abs=1e-8)
    assert 0.3 < table.loc[0, "E_formation"] < 1.2
