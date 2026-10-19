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

Runs real, small, fast calphy calculations rather than fabricating output;
edge cases (a file missing/incomplete, as if a job crashed mid-write) are
exercised by deleting one file from a genuine calphy run's output.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
from ase.build import bulk
from lammpsparser import get_potential_by_name

from phase_diagram_workflows.free_energies.ti_calculator import (
    calc_free_energy_with_calphy,
    gather_calphy_results_detailed,
)


def _real_base_params(mode, temperature):
    return {
        "mode": mode,
        "temperature": temperature,
        "pressure": 0,
        "n_equilibration_steps": 100,
        "n_switching_steps": 100,
        "n_print_steps": 25,
        "equilibration_control": "berendsen",
        "md": {"thermostat_damping": 0.5},
        "tolerance": {"spring_constant": 0.01, "pressure": 0.5},
        "queue": {"cores": 1, "scheduler": "local"},
        "reference_phase": "solid",
        "file_format": "lammps-data",
    }


@pytest.fixture(scope="module")
def real_potential_df():
    pot = get_potential_by_name("1999--Mishin-Y--Al--LAMMPS--ipr1")
    return pot.to_frame().transpose()


@pytest.fixture(scope="module")
def real_structure():
    return bulk("Al", cubic=True).repeat(2)  # 32 atoms: small and fast


@pytest.fixture(scope="module")
def real_ts_calphy_run(real_structure, real_potential_df, tmp_path_factory):
    """A single real, small, fast ts-mode calphy run, reused (read-only) by
    the tests below via copies -- so edge cases (missing files) can be
    exercised on genuine calphy output without rerunning calphy per case."""
    working_directory = str(tmp_path_factory.mktemp("real_ts_run"))
    calc_free_energy_with_calphy(
        input_structure=real_structure,
        potential_df=real_potential_df,
        calphy_parameters=_real_base_params("ts", [700.0, 720.0]),
        working_directory=working_directory,
    )
    return working_directory


@pytest.fixture(scope="module")
def real_fe_calphy_run(real_structure, real_potential_df, tmp_path_factory):
    working_directory = str(tmp_path_factory.mktemp("real_fe_run"))
    calc_free_energy_with_calphy(
        input_structure=real_structure,
        potential_df=real_potential_df,
        calphy_parameters=_real_base_params("fe", 700.0),
        working_directory=working_directory,
    )
    return working_directory


def _copy_of(src_dir, dst_dir):
    shutil.copytree(src_dir, dst_dir)
    return dst_dir


class TestGatherCalphyResultsDetailed:
    """Tests for gather_calphy_results_detailed() edge cases in its own parsing logic.

    Edge cases (a file missing/incomplete, as if a job crashed mid-write) are
    exercised by deleting one file from a genuine calphy run's output,
    rather than fabricating a whole directory of guessed-format files.
    """

    def test_missing_input_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="input_file.yaml"):
                gather_calphy_results_detailed(tmpdir)

    def test_missing_report_yaml(self, real_ts_calphy_run, tmp_path):
        wd = _copy_of(real_ts_calphy_run, str(tmp_path / "run"))
        os.remove(os.path.join(wd, "report.yaml"))

        df = gather_calphy_results_detailed(wd)
        row = df.iloc[0]
        assert bool(row["status"]) is False
        assert row["composition"] is None

    def test_missing_temperature_sweep_falls_back_to_nan_arrays(self, real_ts_calphy_run, tmp_path):
        wd = _copy_of(real_ts_calphy_run, str(tmp_path / "run"))
        os.remove(os.path.join(wd, "temperature_sweep.dat"))

        df = gather_calphy_results_detailed(wd)
        row = df.iloc[0]
        assert len(row["free_energy"]) == len(row["temperature"])
        assert len(row["free_energy_error"]) == len(row["temperature"])
        assert np.all(np.isnan(row["free_energy"]))
        assert np.all(np.isnan(row["free_energy_error"]))

    def test_partial_ts_pair_missing_backward_stops_gathering(self, real_ts_calphy_run, tmp_path):
        wd = _copy_of(real_ts_calphy_run, str(tmp_path / "run"))
        os.remove(os.path.join(wd, "ts.backward_1.dat"))

        df = gather_calphy_results_detailed(wd)
        row = df.iloc[0]
        assert row["forward_energy_diff"] is None
        assert row["backward_energy_diff"] is None

    def test_ts_forward_backward_pairs_are_read(self, real_ts_calphy_run):
        df = gather_calphy_results_detailed(real_ts_calphy_run)
        row = df.iloc[0]
        assert len(row["forward_energy_diff"]) == 1
        assert len(row["backward_energy_diff"]) == 1
        assert row["status"]

    def test_fe_mode_missing_report_gives_nan_error_not_zero(self, real_fe_calphy_run, tmp_path):
        wd = _copy_of(real_fe_calphy_run, str(tmp_path / "run"))
        os.remove(os.path.join(wd, "report.yaml"))

        df = gather_calphy_results_detailed(wd)
        row = df.iloc[0]
        assert np.isnan(row["free_energy"])
        assert np.isnan(row["free_energy_error"])

    def test_fe_mode_with_report_gives_zero_error(self, real_fe_calphy_run):
        df = gather_calphy_results_detailed(real_fe_calphy_run)
        row = df.iloc[0]
        assert not np.isnan(row["free_energy"])
        assert row["free_energy_error"] == 0.0


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
        # the job gives the free energy of the transformation only; free_energy needs the reference (add_reference_free_energy)
        assert row["delta_free_energy"] == pytest.approx(0.0, abs=1.0)
        assert np.isfinite(row["delta_free_energy"])
        assert row["free_energy"] != row["free_energy"]  # NaN until the reference is added
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
        # the job gives the free energy of the transformation only; free_energy needs the reference (add_reference_free_energy)
        assert row["delta_free_energy"] == pytest.approx(0.0, abs=1.0)
        assert np.isfinite(row["delta_free_energy"])
        assert row["free_energy"] != row["free_energy"]  # NaN until the reference is added
        assert row["composition"]["Mg"] == pytest.approx(n_mg_target / n_atoms, abs=1e-6)
