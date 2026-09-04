"""
Integration tests for phase_diagram_workflows.free_energies.ti_calculator.

Real calphy + real LAMMPS runs -- no mocking. Complements
tests/unit/free_energies/test_ti_calculator.py's fast mocked tests (which check
our own dispatch/argument-forwarding contract in isolation) with genuine
end-to-end confidence that mode="composition_scaling" actually drives calphy's
Alchemy class to completion and produces a sane result.

monte_carlo.use_custom_lammps is left False deliberately: True sends LAMMPS a
`fix atom/swap ... noforce yes localE yes` command, where `noforce`/`localE`
are keywords added by the `thermoatoms/lammps` fork (O(1) per-atom energy
evaluation for identity-exchange swaps), not present in plain LAMMPS. CI
installs plain conda-forge `lammps`, so a custom-swap run would hit a hard
LAMMPS parse error ("Illegal fix atom/swap command"), not a graceful skip.
Plain (non-custom) `fix atom/swap` -- exercised here -- is standard LAMMPS
(the MC package) and needs no special build.
"""

import os
import sys

import pytest
from ase.build import bulk

from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy


@pytest.fixture(scope="module")
def al_mg_eam_potential_df():
    """Same real Al-Mg EAM potential used by tests/integration/phase_diagram's
    real-LAMMPS fixture, via lammpsparser + iprpy-data (already a CI dependency,
    see .ci_support/environment.yml)."""
    from lammpsparser import get_potential_by_name

    resource_path = os.path.join(sys.prefix, "share", "iprpy")
    potential_df = get_potential_by_name(
        "1998--Liu-X-Y--Al-Mg--LAMMPS--ipr1", resource_path=resource_path
    )
    potential_df = potential_df.to_frame().transpose()
    potential_df["Config"] = potential_df["Config"].apply(
        lambda cfg: [s if s.endswith("\n") else s + "\n" for s in cfg]
    )
    return potential_df


class TestCompositionScalingRealCalphy:
    """mode="composition_scaling" end-to-end through Alchemy, no executor.

    Small system and step counts, chosen to keep this fast (a few seconds)
    while still exercising the real switching + MC-swap + mass_integration
    pipeline that the unit tests mock out entirely.
    """

    def test_runs_to_completion_with_sane_free_energy(self, al_mg_eam_potential_df, tmp_path):
        structure = bulk("Al", cubic=True).repeat(3)  # 108 atoms
        n_atoms = len(structure)
        n_mg_target = 2

        calphy_parameters = {
            "mode": "composition_scaling",
            "temperature": 300,
            "n_equilibration_steps": 2000,
            "n_switching_steps": 2000,
            "n_print_steps": 0,
            "equilibration_control": "nose-hoover",
            "queue": {"cores": 1, "scheduler": "local"},
            "reference_phase": "solid",
            "file_format": "lammps-data",
            "reference_composition": 0.0,
            "composition_scaling": {
                "output_chemical_composition": {
                    "Al": n_atoms - n_mg_target,
                    "Mg": n_mg_target,
                },
            },
            "monte_carlo": {
                "n_steps": 20,
                "n_swaps": 20,
                "use_custom_lammps": False,
            },
        }

        calculation, df = calc_free_energy_with_calphy(
            input_structure=structure,
            potential_df=al_mg_eam_potential_df,
            calphy_parameters=calphy_parameters,
            working_directory=str(tmp_path),
        )

        row = df.iloc[0]
        assert bool(row["status"]) is True
        assert row["calculation_mode"] == "composition_scaling"
        assert row["reference_phase"] == "solid"
        assert row["free_energy"] == pytest.approx(0.0, abs=1.0)
        assert row["free_energy"] == row["free_energy"]  # not NaN
        assert row["composition"]["Mg"] == pytest.approx(n_mg_target / n_atoms, abs=1e-6)


class TestCompositionScalingWithExternalLmp:
    """mode="composition_scaling" through an externally-managed
    pylammpsmpi session (execution_mode="library", lmp=<SingleNodeExecutor-
    backed LammpsLibrary>) should produce the same kind of sane result as
    the plain path in TestCompositionScalingRealCalphy."""

    def test_runs_to_completion_with_sane_free_energy(
        self, al_mg_eam_potential_df, tmp_path
    ):
        from executorlib import SingleNodeExecutor
        from pylammpsmpi import LammpsLibrary, init_function

        structure = bulk("Al", cubic=True).repeat(4)  # 256 atoms
        n_atoms = len(structure)
        n_mg_target = 5
        cores = 2

        calphy_parameters = {
            "mode": "composition_scaling",
            "temperature": 300,
            "n_equilibration_steps": 5000,
            "n_switching_steps": 5000,
            "n_print_steps": 0,
            "equilibration_control": "nose-hoover",
            "execution_mode": "library",
            "queue": {"cores": cores, "scheduler": "local"},
            "reference_phase": "solid",
            "file_format": "lammps-data",
            "reference_composition": 0.0,
            "composition_scaling": {
                "output_chemical_composition": {
                    "Al": n_atoms - n_mg_target,
                    "Mg": n_mg_target,
                },
            },
            "monte_carlo": {
                "n_steps": 20,
                "n_swaps": 20,
                "use_custom_lammps": False,
            },
        }

        with SingleNodeExecutor(
            block_allocation=True,
            hostname_localhost=True,
            max_workers=1,
            init_function=init_function,
            cache_directory=str(tmp_path / "executorlib_cache"),
            resource_dict={"cores": cores, "cwd": str(tmp_path)},
        ) as executor:
            lmp = LammpsLibrary(cores=cores, executor=executor)
            calculation, df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=al_mg_eam_potential_df,
                calphy_parameters=calphy_parameters,
                working_directory=str(tmp_path),
                lmp=lmp,
            )

        row = df.iloc[0]
        assert bool(row["status"]) is True
        assert row["calculation_mode"] == "composition_scaling"
        assert row["reference_phase"] == "solid"
        assert row["free_energy"] == pytest.approx(0.0, abs=1.0)
        assert row["free_energy"] == row["free_energy"]  # not NaN
        assert row["composition"]["Mg"] == pytest.approx(n_mg_target / n_atoms, abs=1e-6)
