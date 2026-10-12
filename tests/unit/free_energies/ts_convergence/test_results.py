"""
Unit tests for `collect_ts_results` in phase_diagram_workflows.free_energies.ts_convergence.results.

`gather_calphy_results_detailed` is stubbed with a synthetic sweep; the bracket log is the real one, written by
`_record_bracket`, and the calphy input and structure files are small real files in a temporary directory.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

import phase_diagram_workflows.free_energies.ts_convergence.results as results_module
import phase_diagram_workflows.free_energies.ts_convergence.single as single_module
from phase_diagram_workflows.free_energies.ts_convergence.results import collect_ts_results

TEMPERATURE = np.linspace(300.0, 1000.0, 11)


def write_bracket(root, bracket, criterion, with_output=True):
    single_module._record_bracket(str(root), bracket, criterion)
    if not with_output:
        return
    folder = root / f"T_{bracket[0]:.2f}_{bracket[1]:.2f}"
    folder.mkdir(parents=True)
    (folder / "input_file.yaml").write_text(yaml.safe_dump({"calculations": [{
        "pair_coeff": ["* * potential.yace Al Mg"], "n_switching_steps": 140000, "md": {"seed": 6919},
    }]}))
    (folder / "input_structure.data").write_text("Start File for LAMMPS\n\n5324 atoms\n2 atom types\n")


@pytest.fixture(autouse=True)
def stub_gather(monkeypatch):
    def gather(working_directory):
        return pd.DataFrame([{
            "pressure": 0.0,
            "composition": {"Al": 1.0, "Mg": 0.0},
            "temperature": TEMPERATURE,
            "free_energy": -3.6 - 0.001 * (TEMPERATURE - 300.0),
            "free_energy_error": np.zeros_like(TEMPERATURE),
        }])

    monkeypatch.setattr(results_module, "gather_calphy_results_detailed", gather)


def test_finished_phase_is_saved_at_full_resolution_with_what_belongs_to_it(tmp_path):
    root = tmp_path / "run" / "FCC"
    write_bracket(root, (300.0, 1000.0), 0.0005)
    summary = collect_ts_results({"FCC": root}, tolerance=0.001, output_directory=tmp_path / "out")

    assert summary.loc[0, "finished"]
    saved = pd.read_pickle(tmp_path / "out" / "FCC_free_energy.pkl")
    assert list(saved.columns) == ["temperature", "free_energy", "free_energy_error"]
    assert len(saved) == len(TEMPERATURE)
    assert saved.attrs["n_atoms"] == 5324
    assert saved.attrs["composition"] == {"Al": 1.0, "Mg": 0.0}
    assert saved.attrs["criterion"] == 0.0005
    assert (saved.attrs["t_low"], saved.attrs["t_high"]) == (300.0, 1000.0)
    assert saved.attrs["seed"] == 6919
    assert saved.attrs["n_switching_steps"] == 140000


def test_unfinished_phase_is_skipped_and_reported(tmp_path):
    root = tmp_path / "run" / "Gamma"
    write_bracket(root, (300.0, 1000.0), None, with_output=False)   # submitted, no criterion yet
    summary = collect_ts_results({"Gamma": root}, tolerance=float("inf"), output_directory=tmp_path / "out")

    assert not summary.loc[0, "finished"]
    assert summary.loc[0, "pickle"] is None
    assert not (tmp_path / "out" / "Gamma_free_energy.pkl").exists()


def test_bracket_above_the_tolerance_does_not_count(tmp_path):
    root = tmp_path / "run" / "Beta"
    write_bracket(root, (300.0, 900.0), 0.01)
    summary = collect_ts_results({"Beta": root}, tolerance=0.001, output_directory=tmp_path / "out")
    assert not summary.loc[0, "finished"]


def test_several_phases_one_call_and_rerun_overwrites(tmp_path):
    roots = {}
    for name, criterion in [("FCC", 0.0005), ("HCP", 0.0008)]:
        roots[name] = tmp_path / "run" / name
        write_bracket(roots[name], (300.0, 1000.0), criterion)
    for _ in range(2):
        summary = collect_ts_results(roots, tolerance=float("inf"), output_directory=tmp_path / "out")
    assert list(summary["phase"]) == ["FCC", "HCP"]
    assert summary["finished"].all()
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["FCC_free_energy.pkl", "HCP_free_energy.pkl"]
