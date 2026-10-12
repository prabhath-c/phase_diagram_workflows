"""Unit tests for defect_energies.formation_energies; the relaxation is replaced by fake energies, no LAMMPS."""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from phase_diagram_workflows.defect_energies import (
    calculate_formation_energies,
    formation_energies,
    load_defect_results,
    load_phase_defect_results,
    save_defect_results,
)
from phase_diagram_workflows.structures.point_defects import (
    compute_formation_energy,
    discover_atomic_sublattices,
    discover_interstitial_sublattices,
)
from phase_diagram_workflows.utils.paths import archive_directory

OLD_KEYS = [
    "defect_label", "defect_latex", "defect_type", "sublattice_ref", "sublattice_int", "species", "multiplicity",
    "unitcell_multiplicity", "delta_n_Al", "delta_n_Mg", "delta_n_total", "E_defect", "slope", "intercept",
    "slope_mu_Al", "intercept_landau", "dmu_landau_exact", "mu0_Al", "mu0_Mg", "atoms_opt",
]
MU0 = {"Al": -3.6, "Mg": -1.5}
unit_cell = bulk("Al", cubic=True)
atomic = discover_atomic_sublattices(unit_cell)
voids = discover_interstitial_sublattices(unit_cell)


@pytest.fixture
def fake_relaxation(monkeypatch):
    """Energy of a structure: -3.6 eV per Al atom, -1.5 per Mg atom, +0.5 for each atom beyond the pristine count."""
    calls = {}

    def relax(structures, potential_df, working_directory, **kwargs):
        calls.update(kwargs, names=list(structures), working_directory=working_directory)
        rows = {}
        for name, atoms in structures.items():
            symbols = atoms.get_chemical_symbols()
            energy = sum(MU0[s] for s in symbols) + (0.7 if name != "pristine" else 0.0)
            rows[name] = {"structure": atoms, "energy": energy, "status": "done"}
        return pd.DataFrame.from_dict(rows, orient="index")

    monkeypatch.setattr(formation_energies, "relax_structures", relax)
    return calls


def _calculate(kind, orbits, **kwargs):
    return calculate_formation_energies(unit_cell, 2, orbits, kind, pd.DataFrame(), MU0, "Al", "Mg", "wd", **kwargs)


class TestRecords:
    def test_vacancy_has_every_key_of_the_earlier_records_and_the_landau_block(self, fake_relaxation):
        (record,) = _calculate("vacancy", atomic)
        assert set(OLD_KEYS) <= set(record)
        assert record["defect_label"] == "Al_vac_4a" and record["defect_latex"] == "$V_{Al}^{4a}$"
        assert record["defect_type"] == "vacancy" and record["sublattice_ref"] == "Al_4a" and record["sublattice_int"] is None
        assert record["species"] == "Al" and record["multiplicity"] == 32 and record["unitcell_multiplicity"] == 4
        assert (record["delta_n_Al"], record["delta_n_Mg"], record["delta_n_total"]) == (-1, 0, -1)
        assert record["formation"]["landau"] == {"intercept": record["intercept_landau"], "exact": False}
        assert record["dmu_landau_exact"] is False and record["slope"] == 0 and record["slope_mu_Al"] == 1
        assert record["repeat"] == (2, 2, 2) and record["n_atoms"] == 32

    def test_numbers_are_those_of_compute_formation_energy(self, fake_relaxation):
        (record,) = _calculate("vacancy", atomic)
        E_pristine = 32 * MU0["Al"]
        by_hand = compute_formation_energy(E_pristine - MU0["Al"] + 0.7, E_pristine, {"Al": -1}, MU0, "Al", "Mg")
        assert record["E_pristine"] == pytest.approx(E_pristine)
        assert record["intercept"] == pytest.approx(by_hand["intercept"]) == pytest.approx(0.7)
        assert record["intercept_landau"] == pytest.approx(by_hand["landau"]["intercept"])

    def test_substitution_uses_the_other_element_and_is_landau_exact(self, fake_relaxation):
        (record,) = _calculate("substitution", atomic)
        assert record["defect_label"] == "Mg_on_Al_4a" and record["defect_type"] == "substitutional"
        assert record["species"] == "Mg" and record["defect_latex"] == "$Mg_{Al}^{4a}$"
        assert (record["delta_n_Al"], record["delta_n_Mg"], record["delta_n_total"]) == (-1, 1, 0)
        assert record["dmu_landau_exact"] is True and record["slope"] == -1 and record["slope_mu_Al"] == 1

    def test_interstitial_orbits_are_tried_with_every_element(self, fake_relaxation):
        records = _calculate("interstitial", voids)
        assert [r["defect_label"] for r in records] == [f"{el}_i_{o['label']}" for o in voids for el in ("Al", "Mg")]
        assert all(r["sublattice_int"] and r["sublattice_ref"] is None for r in records)
        mg = next(r for r in records if r["defect_label"] == "Mg_i_int_8c")
        assert mg["defect_latex"] == "$Mg_i^{8c}$" and mg["multiplicity"] == 8 * 8 and mg["unitcell_multiplicity"] == 8
        assert (mg["delta_n_Al"], mg["delta_n_Mg"]) == (0, 1)

    def test_elements_can_be_restricted(self, fake_relaxation):
        records = _calculate("interstitial", voids, elements=["Mg"])
        assert {r["species"] for r in records} == {"Mg"}

    def test_the_pristine_supercell_goes_through_the_same_batch_positions_only(self, fake_relaxation):
        _calculate("vacancy", atomic)
        assert fake_relaxation["names"] == ["pristine", "Al_vac_4a"]
        assert fake_relaxation["relax_volume"] is False and fake_relaxation["ftol"] == 1e-8

    def test_unfinished_defects_are_left_out_and_an_unfinished_pristine_gives_nothing(self, monkeypatch):
        def relax(structures, *args, **kwargs):
            return pd.DataFrame({"structure": list(structures.values()), "energy": [-1.0] * len(structures),
                                 "status": ["done", "running"][: len(structures)] + ["running"] * (len(structures) - 2)},
                                index=list(structures))

        monkeypatch.setattr(formation_energies, "relax_structures", relax)
        assert _calculate("interstitial", voids) == []     # pristine is done, the defects are not: nothing to report

        def pristine_pending(structures, *args, **kwargs):
            return pd.DataFrame({"structure": list(structures.values()), "energy": [np.nan] * len(structures),
                                 "status": ["queued"] * len(structures)}, index=list(structures))

        monkeypatch.setattr(formation_energies, "relax_structures", pristine_pending)
        assert _calculate("vacancy", atomic) == []

    def test_a_kind_that_does_not_fit_raises(self, fake_relaxation):
        with pytest.raises(ValueError, match="kind"):
            _calculate("dislocation", atomic)
        with pytest.raises(ValueError, match="atomic orbit"):
            _calculate("vacancy", voids)


class TestStoring:
    def test_zip_of_one_pickle_per_defect_and_a_readable_csv(self, fake_relaxation, tmp_path):
        records = _calculate("interstitial", voids)
        paths = save_defect_results(records, str(tmp_path), "interstitial")
        assert paths["zip"].endswith("interstitial_results.zip") and paths["csv"].endswith("interstitial_results.csv")
        loaded = load_defect_results(paths["zip"])
        assert [r["defect_label"] for r in loaded] == [r["defect_label"] for r in records]
        assert loaded[0]["atoms_opt"] == records[0]["atoms_opt"] and loaded[0]["formation"]["landau"]["exact"] is False
        table = pd.read_csv(paths["csv"])
        assert {"defect_label", "intercept", "intercept_landau", "dmu_landau_exact", "E_pristine"} <= set(table.columns)
        assert "atoms_opt" not in table.columns and len(table) == len(records)

    def test_nothing_to_save_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no records"):
            save_defect_results([], str(tmp_path), "vacancy")

    def test_archive_directory_packs_and_removes(self, tmp_path):
        folder = tmp_path / "cache"
        folder.mkdir()
        (folder / "a.txt").write_text("x")
        archive = archive_directory(folder)
        assert archive.name == "cache.tar.gz" and archive.is_file() and not folder.exists()
        assert archive_directory(folder) is None


class TestLoadPhaseDefectResults:
    def test_reads_the_three_archives_of_a_phase_folder_in_order(self, fake_relaxation, tmp_path):
        for folder, kind, orbits, name in (
            ("Interstitial", "interstitial", voids, "interstitial"),
            ("Vacancy", "vacancy", atomic, "vacancy"),
            ("Substitutional", "substitution", atomic, "substitutional"),
        ):
            (tmp_path / folder).mkdir()
            save_defect_results(_calculate(kind, orbits), str(tmp_path / folder), name)
        records = load_phase_defect_results(str(tmp_path))
        assert [r["defect_type"] for r in records] == ["vacancy", "substitutional"] + ["interstitial"] * (2 * len(voids))

    def test_a_missing_archive_is_skipped_and_no_archive_at_all_raises(self, fake_relaxation, tmp_path):
        (tmp_path / "Vacancy").mkdir()
        with pytest.raises(FileNotFoundError, match="all_"):
            load_phase_defect_results(str(tmp_path))
        save_defect_results(_calculate("vacancy", atomic), str(tmp_path / "Vacancy"), "vacancy")
        assert len(load_phase_defect_results(str(tmp_path))) == 1
