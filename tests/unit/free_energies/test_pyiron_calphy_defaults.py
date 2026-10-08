import pytest

from phase_diagram_workflows.free_energies.pyiron_calphy_defaults import pyiron_atomistics_calphy_parameters


def test_defaults_match_pyiron_atomistics():
    parameters = pyiron_atomistics_calphy_parameters()
    assert parameters["n_equilibration_steps"] == 15000
    assert parameters["n_switching_steps"] == 25000
    assert parameters["equilibration_control"] == "nose-hoover"
    assert parameters["pressure"] == 0 and parameters["npt"] is True
    assert parameters["md"]["thermostat_damping"] == 0.5
    assert parameters["tolerance"] == {
        "lattice_constant": 0.0002, "spring_constant": 0.01, "solid_fraction": 0.7, "liquid_fraction": 0.05, "pressure": 0.5,
    }
    assert parameters["berendsen"] == {"thermostat_damping": 100.0, "barostat_damping": 100.0}


def test_per_calculation_keys_are_left_to_the_caller():
    parameters = pyiron_atomistics_calphy_parameters()
    assert not {"mode", "temperature", "reference_phase"} & set(parameters)


def test_every_call_is_an_independent_deep_copy():
    first = pyiron_atomistics_calphy_parameters()
    first["md"]["seed"] = 1
    first["tolerance"]["pressure"] = 9.0
    first["temperature"] = [300, 1000]
    second = pyiron_atomistics_calphy_parameters()
    assert "seed" not in second["md"]
    assert second["tolerance"]["pressure"] == 0.5
    assert "temperature" not in second


def test_accepted_by_calphy():
    calphy = pytest.importorskip("calphy")
    parameters = pyiron_atomistics_calphy_parameters()
    parameters.update(mode="ts", temperature=[300, 320], reference_phase="solid")
    calculation = calphy.Calculation.model_validate(
        {**parameters, "element": ["Al"], "mass": [26.98], "pair_style": ["eam/alloy"], "pair_coeff": ["* * x.eam Al"]}
    )
    assert calculation.tolerance.solid_fraction == 0.7 and calculation.spring_constants is None
