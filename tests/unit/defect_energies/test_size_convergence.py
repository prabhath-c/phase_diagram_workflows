"""Unit tests for defect_energies.size_convergence that need no LAMMPS: defect building, cores, table, resume."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from phase_diagram_workflows.defect_energies import size_convergence
from phase_diagram_workflows.structures.point_defects import discover_atomic_sublattices, discover_interstitial_sublattices
from phase_diagram_workflows.utils.executor_spec import ExecutorSpec
from phase_diagram_workflows.defect_energies.size_convergence import (
    RESULT_FILE,
    _build_defect,
    _min_width,
    converge_formation_energy,
    cores_for_size,
    select_converged_size,
    summarize_size_convergence,
    load_size_convergence,
)


def _write_result(folder, repeat, n_atoms, E_formation):
    os.makedirs(folder, exist_ok=True)
    result = {
        "repeat": repeat, "n_atoms": n_atoms, "min_width_A": 4.0 * repeat[0], "E_pristine": -3.0 * n_atoms,
        "E_pristine_per_atom": -3.0, "max_force_pristine": 0.0, "E_defect": -3.0 * n_atoms + 1.0,
        "E_formation": E_formation, "cores": 1, "defect_structure": bulk("Al"),
    }
    with open(os.path.join(folder, RESULT_FILE), "wb") as handle:
        pickle.dump(result, handle)


class TestBuildDefect:
    unit_cell = bulk("Al", cubic=True)

    def test_vacancy(self):
        atoms, delta_n = _build_defect(self.unit_cell.repeat(2), self.unit_cell, {"type": "vacancy", "atom_index": 0})
        assert len(atoms) == 31 and delta_n == {"Al": -1}

    def test_substitution(self):
        atoms, delta_n = _build_defect(
            self.unit_cell.repeat(2), self.unit_cell, {"type": "substitution", "atom_index": 1, "to_element": "Mg"}
        )
        assert len(atoms) == 32 and atoms.get_chemical_symbols().count("Mg") == 1
        assert delta_n == {"Mg": 1, "Al": -1}

    def test_interstitial_sits_at_the_given_position(self):
        position = [2.025, 0.0, 0.0]
        atoms, delta_n = _build_defect(
            self.unit_cell.repeat(2), self.unit_cell, {"type": "interstitial", "position": position, "element": "Mg"}
        )
        assert len(atoms) == 33 and delta_n == {"Mg": 1}
        assert np.allclose(atoms.positions[-1], position)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            _build_defect(self.unit_cell.repeat(2), self.unit_cell, {"type": "dislocation"})


class SlurmClusterExecutor:
    """Records the jobs; defined at module level because an ExecutorSpec has to be picklable, and named like
    the real class so that it counts as one that can run on without this process."""

    submitted = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def submit(self, fn, *args, resource_dict):
        type(self).submitted.append((args[2], resource_dict))

    def shutdown(self, wait):
        self.waited = wait


class TestHelpers:
    def test_cores_for_size(self):
        assert cores_for_size(32) == 1 and cores_for_size(1499) == 1
        assert cores_for_size(2048) == 8 and cores_for_size(5324) == 16 and cores_for_size(10176) == 16   # powers of two, capped
        assert all(cores_for_size(n) in (1, 2, 4, 8, 16) for n in range(1, 20000, 137))

    def test_min_width_of_orthorhombic_and_hexagonal_cells(self):
        assert _min_width(np.diag([4.0, 5.0, 6.0])) == pytest.approx(4.0)
        hexagonal = bulk("Mg").cell.array * 3
        assert _min_width(hexagonal) == pytest.approx(3 * 3.21 * np.sin(np.radians(60)), rel=0.02)


class TestLoadSizeConvergence:
    def test_table_sorted_with_delta_and_edge_flag(self, tmp_path):
        _write_result(tmp_path / "repeat_3x3x3", (3, 3, 3), 108, 0.60)
        _write_result(tmp_path / "repeat_2x2x2", (2, 2, 2), 32, 0.70)
        table = load_size_convergence(str(tmp_path), rcut=3.0)
        assert list(table["n_atoms"]) == [32, 108]
        assert table.loc[1, "delta_meV"] == pytest.approx(-100.0)
        assert list(table["edge_clears_2rcut"]) == [True, True]
        assert (tmp_path / "size_convergence.csv").is_file()

    def test_empty_folder_gives_empty_table(self, tmp_path):
        assert load_size_convergence(str(tmp_path / "none")).empty


class TestConvergeFormationEnergy:
    unit_cell = bulk("Al", cubic=True)
    orbit = discover_atomic_sublattices(unit_cell)[0]
    common = dict(potential_df=pd.DataFrame(), mu0={"Al": -3.0, "Mg": -1.5}, pinned_element="Al", swept_element="Mg")

    def test_finished_sizes_are_skipped(self, tmp_path, monkeypatch):
        _write_result(tmp_path / "repeat_2x2x2", (2, 2, 2), 32, 0.7)
        ran = []
        monkeypatch.setattr(size_convergence, "_formation_energy_at_size", lambda *a: ran.append(a[2]))
        converge_formation_energy(self.unit_cell, self.orbit, "vacancy", repeats=[2, 3], working_directory=str(tmp_path), **self.common)
        assert ran == [(3, 3, 3)]

    def test_each_size_is_submitted_with_its_own_cores(self, tmp_path, monkeypatch):
        submitted = []
        SlurmClusterExecutor.submitted = submitted
        converge_formation_energy(
            self.unit_cell, self.orbit, "vacancy", repeats=[2, 11], working_directory=str(tmp_path),
            executor_spec=ExecutorSpec(SlurmClusterExecutor, {"resource_dict": {"queue": "cmmg"}, "cache_directory": "c"}), **self.common,
        )
        assert [s[0] for s in submitted] == [(2, 2, 2), (11, 11, 11)]
        assert submitted[0][1]["threads_per_core"] == 1 and submitted[1][1]["threads_per_core"] == 16   # 5324 atoms
        assert all(s[1]["queue"] == "cmmg" and s[1]["cores"] == 1 for s in submitted)
        assert len({s[1]["cache_key"] for s in submitted}) == 2

    def test_duplicate_sizes_a_kind_that_does_not_fit_and_a_wrong_site_raise(self, tmp_path):
        with pytest.raises(ValueError, match="twice"):
            converge_formation_energy(self.unit_cell, self.orbit, "vacancy", repeats=[2, (2, 2, 2)], working_directory=str(tmp_path), **self.common)
        with pytest.raises(ValueError, match="interstitial"):
            converge_formation_energy(self.unit_cell, self.orbit, "interstitial", element="Mg", repeats=[2], working_directory=str(tmp_path), **self.common)
        wrong = {**self.orbit, "atom_indices": [0], "multiplicity": 3}   # claims 3 sites per cell, so 24 in 2x2x2: not what is there
        with pytest.raises(ValueError, match="tiled"):
            converge_formation_energy(self.unit_cell, wrong, "vacancy", repeats=[2], working_directory=str(tmp_path), **self.common)

    def test_the_interstitial_site_is_checked_too(self, tmp_path, monkeypatch):
        void = next(o for o in discover_interstitial_sublattices(self.unit_cell) if o["label"] == "int_8c")
        ran = []
        monkeypatch.setattr(size_convergence, "_formation_energy_at_size", lambda *a: ran.append(a[2]))
        converge_formation_energy(
            self.unit_cell, void, "interstitial", element="Mg", repeats=[2], working_directory=str(tmp_path / "ok"), **self.common
        )
        assert ran == [(2, 2, 2)]
        wrong = {**void, "multiplicity": 99}      # the tiled orbit would have 99 x 8 sites, not what the cell has
        with pytest.raises(ValueError, match="tiled"):
            converge_formation_energy(
                self.unit_cell, wrong, "interstitial", element="Mg", repeats=[2], working_directory=str(tmp_path / "bad"), **self.common
            )


class TestSummarizeSizeConvergence:
    def _results(self, tmp_path, energies):
        sizes = [((2, 2, 2), 32), ((3, 3, 3), 108), ((4, 4, 4), 256)]
        for (repeat, n_atoms), energy in zip(sizes, energies):
            _write_result(tmp_path / "ptmp" / size_convergence._repeat_name(repeat), repeat, n_atoms, energy)
        (tmp_path / "notebook").mkdir(exist_ok=True)

    def test_writes_plot_table_and_converged_size(self, tmp_path):
        import json

        self._results(tmp_path, [0.20, 0.1305, 0.1300])
        table, converged = summarize_size_convergence(
            str(tmp_path / "ptmp"), "size_convergence_x", str(tmp_path / "notebook"), tolerance_meV=5.0, rcut=1.0,
            extra={"phase": "FCC"},
        )
        assert converged["repeat"] == (3, 3, 3) and converged["phase"] == "FCC"
        assert json.load(open(tmp_path / "notebook" / "converged_size.json"))["n_atoms"] == 108
        assert (tmp_path / "notebook" / "size_convergence_x.png").is_file()
        assert "defect_structure" not in pd.read_pickle(tmp_path / "notebook" / "size_convergence_x.pkl").columns

    def test_not_converged_removes_a_stale_converged_size(self, tmp_path):
        self._results(tmp_path, [0.30, 0.20, 0.10])
        (tmp_path / "notebook" / "converged_size.json").write_text("{}")
        _, converged = summarize_size_convergence(str(tmp_path / "ptmp"), "x", str(tmp_path / "notebook"))
        assert converged is None and not (tmp_path / "notebook" / "converged_size.json").exists()

    def test_no_results_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no results"):
            summarize_size_convergence(str(tmp_path), "x", str(tmp_path))


class TestSelectConvergedSize:
    @staticmethod
    def _table(energies, widths=None):
        table = pd.DataFrame({
            "repeat": [(2, 2, 2), (3, 3, 3), (4, 4, 4), (6, 6, 6)], "n_atoms": [32, 108, 256, 864], "E_formation": energies,
        })
        table["delta_meV"] = table["E_formation"].diff() * 1000
        if widths is not None:
            table["edge_clears_2rcut"] = widths
        return table

    def test_first_size_within_the_tolerance_of_the_largest(self):
        chosen = select_converged_size(self._table([0.200, 0.140, 0.1332, 0.1330]), tolerance_meV=5.0)
        assert chosen["repeat"] == (4, 4, 4) and chosen["n_atoms"] == 256
        assert chosen["distance_to_largest_meV"] == pytest.approx(0.2)

    def test_the_default_tolerance_is_1_meV(self):
        assert select_converged_size(self._table([0.200, 0.1345, 0.1335, 0.1330]))["n_atoms"] == 256   # 108 atoms is 1.5 meV away

    def test_narrow_cells_are_skipped(self):
        table = self._table([0.1332, 0.1331, 0.1330, 0.1330], widths=[False, False, True, True])
        assert select_converged_size(table)["n_atoms"] == 256

    def test_the_largest_size_is_the_reference_and_not_a_candidate(self):
        assert select_converged_size(self._table([0.30, 0.25, 0.20, 0.15]), tolerance_meV=5.0) is None

    def test_one_size_cannot_be_judged(self):
        assert select_converged_size(self._table([0.1, 0.1, 0.1, 0.1]).iloc[:1]) is None


class TestPlot:
    @staticmethod
    def _table(clears):
        return pd.DataFrame({
            "n_atoms": [32, 108, 256], "E_formation": [0.20, 0.14, 0.13], "E_pristine_per_atom": [-3.6] * 3,
            "min_width_A": [8.087, 12.13, 16.17], "edge_clears_2rcut": clears,
        })

    def test_two_panels_and_the_cutoff_as_a_vertical_line_where_the_width_reaches_2_rcut(self):
        from phase_diagram_workflows.defect_energies import plot_size_convergence

        figure = plot_size_convergence(self._table([False, True, True]), title="x", rcut=5.5, tolerance_meV=1.0)
        assert len(figure.axes) == 2 and figure.axes[1].get_yscale() == "log"
        for ax in figure.axes:
            (line,) = [l for l in ax.lines if len(set(l.get_xdata())) == 1]
            assert line.get_xdata()[0] == pytest.approx(32 * (11.0 / 8.087) ** 3)     # ~80 atoms, between 32 and 108
        assert not any(ax.patches for ax in figure.axes)

    def test_no_line_when_every_size_is_wide_enough_or_none_is(self):
        from phase_diagram_workflows.defect_energies import plot_size_convergence

        for clears in ([True, True, True], [False, False, False]):
            figure = plot_size_convergence(self._table(clears), rcut=5.5)
            assert not any(len(set(l.get_xdata())) == 1 for ax in figure.axes for l in ax.lines)
