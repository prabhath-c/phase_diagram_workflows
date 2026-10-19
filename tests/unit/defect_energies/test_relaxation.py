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

    def test_cores_and_oversubscribe_reach_the_task(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(relaxation, "run_nested_batch", lambda **kw: captured.update(kw))
        for oversubscribe in (True, False):
            relax_structures(
                {"a": bulk("Al")}, pd.DataFrame(), str(tmp_path / str(oversubscribe)),
                executor_spec=ExecutorSpec(object, {}), threads_per_structure=4, oversubscribe=oversubscribe,
            )
            assert captured["task_fn"].keywords == {"anisotropic": False, "relax_options": None, "cores": 4, "oversubscribe": oversubscribe}

    def test_anisotropic_reaches_the_task(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(relaxation, "run_nested_batch", lambda **kw: captured.update(kw))
        relax_structures({"a": bulk("Mg")}, pd.DataFrame(), str(tmp_path), executor_spec=ExecutorSpec(object, {}), anisotropic=True)
        assert captured["task_fn"].keywords["anisotropic"] is True

    def test_relax_options_reach_the_task_and_are_checked_before_anything_is_submitted(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(relaxation, "run_nested_batch", lambda **kw: captured.update(kw))
        relax_structures({"a": bulk("Mg")}, pd.DataFrame(), str(tmp_path / "ok"), executor_spec=ExecutorSpec(object, {}),
                         relax_options={"etol": 1e-12, "vmax": 0.001})
        assert captured["task_fn"].keywords["relax_options"] == {"etol": 1e-12, "vmax": 0.001}
        with pytest.raises(ValueError, match="relax_options"):
            relax_structures({"a": bulk("Mg")}, pd.DataFrame(), str(tmp_path / "bad"), relax_options={"etool": 1e-12})
        assert not (tmp_path / "bad" / "a").exists()          # nothing was marked as submitted


class TestRelaxAtoms:
    @pytest.fixture
    def libcalculator(self, monkeypatch):
        import atomistics.calculators.lammps.libcalculator as libcalculator
        calls = {"volume": [], "positions": []}
        monkeypatch.setattr(libcalculator, "optimize_positions_and_volume_with_lammpslib", lambda **kw: calls["volume"].append(kw) or kw["structure"])
        monkeypatch.setattr(libcalculator, "optimize_positions_with_lammpslib", lambda **kw: calls["positions"].append(kw) or kw["structure"])
        return calls

    def test_cores_and_oversubscribe_reach_atomistics(self, libcalculator):
        relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), True, 1e-6, cores=4, oversubscribe=True)
        assert libcalculator["volume"][0]["cores"] == 4 and libcalculator["volume"][0]["oversubscribe"] is True

    def test_positions_only_never_touches_the_cell_even_if_anisotropic(self, libcalculator):
        relaxation.relax_atoms(bulk("Mg"), pd.DataFrame(), False, 1e-6, anisotropic=True)
        assert len(libcalculator["positions"]) == 1 and not libcalculator["volume"]

    def test_isotropic_is_atomistics_own_volume_relaxation(self, libcalculator):
        relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), True, 1e-6)
        assert len(libcalculator["volume"]) == 1

    def test_anisotropic_goes_through_the_six_component_box_relax(self, libcalculator, monkeypatch):
        seen = {}
        monkeypatch.setattr(relaxation, "_optimize_anisotropic", lambda atoms, potential_df, ftol, options, **kw: seen.update(ftol=ftol, options=options, kw=kw) or atoms)
        relaxation.relax_atoms(bulk("Mg"), pd.DataFrame(), True, 1e-8, anisotropic=True, cores=2, oversubscribe=True, relax_options={"etol": 1e-10})
        assert seen == {"ftol": 1e-8, "options": {"etol": 1e-10}, "kw": {"cores": 2, "oversubscribe": True}} and not libcalculator["volume"]

    def test_only_the_options_given_reach_atomistics_so_its_defaults_apply_to_the_rest(self, libcalculator):
        relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), True, 1e-8)
        relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), True, 1e-8, relax_options={"etol": 1e-12, "pressure": 100.0, "vmax": 0.001})
        relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), False, 1e-8, relax_options={"maxiter": 5, "min_style": "fire"})
        plain, with_options = libcalculator["volume"]
        assert set(plain) == {"structure", "potential_dataframe", "ftol", "cores", "oversubscribe"}
        assert {key: with_options[key] for key in ("etol", "pressure", "vmax")} == {"etol": 1e-12, "pressure": 100.0, "vmax": 0.001}
        assert libcalculator["positions"][0]["maxiter"] == 5 and libcalculator["positions"][0]["min_style"] == "fire"

    def test_unknown_options_and_cell_options_without_a_cell_relaxation_are_refused(self, libcalculator):
        with pytest.raises(ValueError, match="etool"):
            relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), True, 1e-8, relax_options={"etool": 1e-12})
        with pytest.raises(ValueError, match="vmax"):
            relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), False, 1e-8, relax_options={"vmax": 0.001})
        with pytest.raises(ValueError, match="argument of its own"):
            relaxation.relax_atoms(bulk("Al"), pd.DataFrame(), True, 1e-8, relax_options={"ftol": 1e-8})

    @pytest.fixture
    def sent(self, monkeypatch):
        import atomistics.calculators.lammps.helpers as helpers
        sent = {}
        monkeypatch.setattr(helpers, "lammps_run", lambda **kw: sent.update(kw) or object())
        monkeypatch.setattr(helpers, "lammps_get_structure", lambda **kw: kw["structure"])
        monkeypatch.setattr(helpers, "lammps_shutdown", lambda **kw: None)
        return sent

    def test_the_commands_make_the_box_triclinic_and_relax_all_six_components_with_the_atomistics_defaults(self, sent):
        import inspect
        from atomistics.calculators.lammps.libcalculator import optimize_positions_and_volume_with_lammpslib
        defaults = {name: parameter.default for name, parameter in inspect.signature(optimize_positions_and_volume_with_lammpslib).parameters.items()}
        relaxation._optimize_anisotropic(bulk("Mg"), pd.DataFrame(), 1e-8, {})
        lines = sent["input_template"].splitlines()
        assert lines[0] == "change_box all triclinic"
        assert lines[1] == "fix ensemble all box/relax x 0.0 y 0.0 z 0.0 xy 0.0 xz 0.0 yz 0.0"
        assert f"thermo {defaults['thermo']}" in lines and f"min_style {defaults['min_style']}" in lines
        assert f"minimize {defaults['etol']} 1e-08 {defaults['maxiter']} {defaults['maxeval']}" in lines

    def test_every_option_given_reaches_the_commands(self, sent):
        options = {"min_style": "fire", "etol": 1e-12, "maxiter": 50, "maxeval": 500, "thermo": 5, "pressure": 100.0, "vmax": 0.001}
        relaxation._optimize_anisotropic(bulk("Mg"), pd.DataFrame(), 1e-8, options)
        lines = sent["input_template"].splitlines()
        assert lines[1] == "fix ensemble all box/relax x 100.0 y 100.0 z 100.0 xy 0.0 xz 0.0 yz 0.0 vmax 0.001"
        assert "min_style fire" in lines and "thermo 5" in lines and "minimize 1e-12 1e-08 50 500" in lines

    def test_a_pressure_list_is_taken_component_wise(self, sent):
        relaxation._optimize_anisotropic(bulk("Mg"), pd.DataFrame(), 1e-8, {"pressure": [1.0, 2.0, 3.0, None, None, None]})
        assert sent["input_template"].splitlines()[1] == "fix ensemble all box/relax x 1.0 y 2.0 z 3.0"
