"""Unit tests for defect_energies.relaxation that need no LAMMPS: naming, validation, table and resume logic."""

import os
import pickle

import pandas as pd
import pytest
from ase.build import bulk

from phase_diagram_workflows.defect_energies import relaxation
from phase_diagram_workflows.utils.executor_spec import ExecutorSpec
from phase_diagram_workflows.defect_energies.relaxation import (
    RESULT_FILE,
    _as_named_structures,
    load_relaxed_structures,
    relax_structures,
)


def _write_result(folder, name, atoms, energy):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, RESULT_FILE), "wb") as handle:
        pickle.dump({"name": name, "structure": atoms, "energy": energy, "relax_volume": True}, handle)


class TestNaming:
    def test_dict_list_and_dataframe(self):
        atoms = bulk("Al")
        assert list(_as_named_structures({"a": atoms})) == ["a"]
        assert list(_as_named_structures([atoms, atoms])) == ["structure_0", "structure_1"]
        assert list(_as_named_structures(pd.DataFrame({"structure": [atoms]}, index=["x"]))) == ["x"]

    def test_materials_project_style_dataframe(self):
        atoms = bulk("Al")
        df = pd.DataFrame({"structure": [{"pymatgen": "dict"}], "structure_ase": [atoms], "material_id_underscore": ["mp_134"]})
        named = _as_named_structures(df, atoms_col="structure_ase", name_col="material_id_underscore")
        assert list(named) == ["mp_134"] and named["mp_134"] is atoms

    def test_duplicate_names_raise(self):
        df = pd.DataFrame({"structure": [bulk("Al"), bulk("Al")], "n": ["a", "a"]})
        with pytest.raises(ValueError, match="unique"):
            _as_named_structures(df, name_col="n")

    def test_dataframe_without_structure_column_raises(self):
        with pytest.raises(ValueError, match="structure"):
            _as_named_structures(pd.DataFrame({"atoms": [bulk("Al")]}))

    def test_name_with_path_separator_raises(self):
        with pytest.raises(ValueError, match="folder name"):
            _as_named_structures({"a/b": bulk("Al")})


class TestLoadRelaxedStructures:
    def test_table_columns_and_files(self, tmp_path):
        atoms = bulk("Al", cubic=True)
        _write_result(tmp_path / "fcc", "fcc", atoms, -14.0)
        table = load_relaxed_structures(str(tmp_path))
        assert table.loc["fcc", "n_atoms"] == 4
        assert table.loc["fcc", "energy_per_atom"] == pytest.approx(-3.5)
        assert table.loc["fcc", "status"] == "done"
        assert (tmp_path / "relaxed_structures.csv").is_file()
        assert pd.read_pickle(tmp_path / "relaxed_structures.pkl").loc["fcc", "structure"] == atoms

    def test_names_gives_pending_rows(self, tmp_path):
        _write_result(tmp_path / "a", "a", bulk("Al"), -3.0)
        table = load_relaxed_structures(str(tmp_path), names=["a", "b"])
        assert list(table.index) == ["a", "b"]
        assert list(table["status"]) == ["done", "pending"]

    def test_missing_folder_gives_empty_table(self, tmp_path):
        assert load_relaxed_structures(str(tmp_path / "nothing")).empty


class TestRelaxStructures:
    def test_finished_structures_are_not_run_again(self, tmp_path, monkeypatch):
        _write_result(tmp_path / "a", "a", bulk("Al"), -3.0)
        calls = []
        monkeypatch.setattr(relaxation, "_relax_and_save", lambda item, *args: calls.append(item[0]))
        relax_structures({"a": bulk("Al"), "b": bulk("Al")}, pd.DataFrame(), str(tmp_path))
        assert calls == ["b"]

    def test_executor_gets_one_batch_and_nothing_else(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(relaxation, "run_nested_batch", lambda **kw: captured.update(kw))
        table = relax_structures(
            {"a": bulk("Al"), "b": bulk("Al")}, pd.DataFrame(), str(tmp_path),
            executor_spec=ExecutorSpec(object, {"cache_directory": "c", "resource_dict": {"queue": "q"}}),
            structures_in_parallel=10, threads_per_structure=3,
        )
        assert [item[0] for item in captured["items"]] == ["a", "b"]
        assert captured["inner_max_workers"] == 2 and captured["wait"] is True   # a plain class cannot detach
        assert captured["inner_resource_dict"] == {"cores": 1, "threads_per_core": 3}
        assert captured["outer_resource_dict"] == {"queue": "q"} and captured["cache_directory"] == "c"
        assert list(table["status"]) == ["queued", "queued"]   # submitted, nothing started
