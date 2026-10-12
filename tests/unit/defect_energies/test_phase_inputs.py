"""Unit tests for defect_energies.phase_inputs.load_phase_inputs and utils.paths.scratch_directory."""

import pickle

import pandas as pd
import pytest
from ase.build import bulk

from phase_diagram_workflows.defect_energies import load_phase_inputs
from phase_diagram_workflows.utils.paths import scratch_directory


@pytest.fixture
def pickles(tmp_path):
    cell = bulk("Al", cubic=True)
    table = pd.DataFrame({"structure": [cell]}, index=["FCC"])
    with open(tmp_path / "pristine.pkl", "wb") as handle:
        pickle.dump({"table": table, "mu0_Al": -3.6, "mu0_Mg": -1.5, "potential": "p.yace"}, handle)
    with open(tmp_path / "sublattices.pkl", "wb") as handle:
        pickle.dump({"sublattices": {"FCC": {"atomic": [{"label": "Al_4a"}], "interstitial": []}}, "potential": "p.yace"}, handle)
    return tmp_path / "pristine.pkl", tmp_path / "sublattices.pkl"


def test_reads_cell_mu0_for_every_element_and_orbits(pickles):
    inputs = load_phase_inputs(*pickles, "FCC", potential="p.yace")
    assert len(inputs.unit_cell) == 4 and inputs.mu0 == {"Al": -3.6, "Mg": -1.5}
    assert inputs.atomic == [{"label": "Al_4a"}] and inputs.interstitial == [] and inputs.potential == "p.yace"


def test_a_pickle_from_another_potential_is_refused(pickles):
    with pytest.raises(ValueError, match="other.yace"):
        load_phase_inputs(*pickles, "FCC", potential="other.yace")


def test_unknown_phase_raises(pickles):
    with pytest.raises(KeyError):
        load_phase_inputs(*pickles, "Gamma")


def test_scratch_directory_mirrors_the_path_below_home(tmp_path):
    home = tmp_path / "home"
    notebook = home / "work" / "phase"
    notebook.mkdir(parents=True)
    assert scratch_directory(notebook, "/scratch", "run", home=home) == __import__("pathlib").Path("/scratch/work/phase/run")
    with pytest.raises(ValueError):
        scratch_directory(tmp_path / "elsewhere", "/scratch", "run", home=home)
