"""
Integration tests for phase_diagram_workflows.free_energies.ti_calculator.

Runs real, small, fast calphy calculations rather than fabricating output;
edge cases (a file missing/incomplete, as if a job crashed mid-write) are
exercised by deleting one file from a genuine calphy run's output.
"""

import os
import shutil
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

