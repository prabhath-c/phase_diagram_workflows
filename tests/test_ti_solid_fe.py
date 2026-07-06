"""
Unit tests for calphy workflows.

Tests validation functions and main workflow with mocked external dependencies.
"""

import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.build import bulk
from lammpsparser import get_potential_by_name

from phase_diagram_workflows.free_energies.ti_helpers import (
    _validate_input_structure,
    _validate_potential_df,
    _validate_calphy_parameters,
    _working_directory_context,
)
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


class TestValidateInputStructure:
    """Tests for _validate_input_structure()"""
    
    def test_valid_structure(self):
        """Test that valid structure passes without error"""
        structure = bulk('Al', cubic=True)
        _validate_input_structure(structure)  # Should not raise
    
    def test_empty_structure_raises_error(self):
        """Test that empty structure raises ValueError"""
        structure = Atoms()
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_input_structure(structure)


class TestValidatePotentialDf:
    """Tests for _validate_potential_df()"""
    
    def test_valid_potential_df(self):
        """Test that valid DataFrame passes without error"""
        df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        _validate_potential_df(df)  # Should not raise
    
    def test_missing_Species_column_raises_error(self):
        """Test that missing Species column raises ValueError"""
        df = pd.DataFrame({
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        with pytest.raises(ValueError, match="Species"):
            _validate_potential_df(df)
    
    def test_missing_Config_column_raises_error(self):
        """Test that missing Config column raises ValueError"""
        df = pd.DataFrame({
            'Species': [['Al']]
        })
        with pytest.raises(ValueError, match="Config"):
            _validate_potential_df(df)


class TestValidateCalphyParameters:
    """Tests for _validate_calphy_parameters()"""
    
    def test_valid_parameters(self):
        """Test that valid parameters pass without error"""
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        _validate_calphy_parameters(params)  # Should not raise
    
    def test_missing_mode_raises_error(self):
        """Test that missing mode raises ValueError"""
        params = {
            'temperature': 300,
            'reference_phase': 'solid'
        }
        with pytest.raises(ValueError, match="mode"):
            _validate_calphy_parameters(params)
    
    def test_missing_temperature_raises_error(self):
        """Test that missing temperature raises ValueError"""
        params = {
            'mode': 'fe',
            'reference_phase': 'solid'
        }
        with pytest.raises(ValueError, match="temperature"):
            _validate_calphy_parameters(params)
    
    def test_missing_reference_phase_raises_error(self):
        """Test that missing reference_phase raises ValueError"""
        params = {
            'mode': 'fe',
            'temperature': 300
        }
        with pytest.raises(ValueError, match="reference_phase"):
            _validate_calphy_parameters(params)
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        params = {
            'mode': 'invalid',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        with pytest.raises(ValueError, match="mode must be"):
            _validate_calphy_parameters(params)
    
    def test_invalid_reference_phase_raises_error(self):
        """Test that invalid reference_phase raises ValueError"""
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'gas'
        }
        with pytest.raises(ValueError, match="reference_phase must be"):
            _validate_calphy_parameters(params)
    
    def test_valid_ts_mode(self):
        """Test that 'ts' mode is accepted"""
        params = {
            'mode': 'ts',
            'temperature': [300, 600],
            'reference_phase': 'solid'
        }
        _validate_calphy_parameters(params)  # Should not raise
    
    def test_valid_liquid_phase(self):
        """Test that 'liquid' phase is accepted"""
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'liquid'
        }
        _validate_calphy_parameters(params)  # Should not raise


class TestWorkingDirectoryContext:
    """Tests for _working_directory_context()"""
    
    def test_context_manager_changes_directory(self):
        """Test that context manager changes to specified directory"""
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            with _working_directory_context(tmpdir):
                assert os.getcwd() == tmpdir
            # Should be back to original directory
            assert os.getcwd() == original_dir
    
    def test_context_manager_creates_directory(self):
        """Test that context manager creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'subdir', 'nested')
            assert not os.path.exists(new_dir)
            
            with _working_directory_context(new_dir):
                assert os.path.exists(new_dir)
                assert os.getcwd() == new_dir
    
    def test_context_manager_restores_directory_on_error(self):
        """Test that context manager restores directory even on error"""
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with _working_directory_context(tmpdir):
                    raise RuntimeError("Test error")
            except RuntimeError:
                pass
            
            # Should be back to original directory despite error
            assert os.getcwd() == original_dir


class TestCalcFreeEnergyWithCalphydIntegration:
    """Integration tests for main workflow with mocking"""
    
    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_main_function_calls_correct_sequence(
        self, mock_build_config, mock_gather, mock_run_calphy
    ):
        """Test that main function calls helper functions in correct sequence"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy
        
        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})
        
        # Setup inputs
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call main function
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                user_dict={}
            )
            
            # Verify correct calls were made
            mock_build_config.assert_called_once()
            mock_run_calphy.assert_called_once()
            mock_gather.assert_called_once()
            
            # Verify return values
            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)


class TestInputValidationErrors:
    """Test main workflow input validation"""

    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_invalid_structure_raises_error(self, mock_build_config):
        """Test that invalid structure raises ValueError"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        structure = Atoms()  # Empty - invalid
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Input validation failed"):
                calc_free_energy_with_calphy(
                    input_structure=structure,
                    potential_df=potential_df,
                    calphy_parameters=params,
                    working_directory=tmpdir,
                    user_dict={}
                )

    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_invalid_potential_raises_error(self, mock_build_config):
        """Test that invalid potential DataFrame raises ValueError"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({'NotSpecies': [['Al']]})  # Wrong column
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Input validation failed"):
                calc_free_energy_with_calphy(
                    input_structure=structure,
                    potential_df=potential_df,
                    calphy_parameters=params,
                    working_directory=tmpdir,
                    user_dict={}
                )

    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_invalid_parameters_raises_error(self, mock_build_config):
        """Test that invalid parameters raises ValueError"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {'mode': 'invalid'}  # Missing temperature, reference_phase

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Input validation failed"):
                calc_free_energy_with_calphy(
                    input_structure=structure,
                    potential_df=potential_df,
                    calphy_parameters=params,
                    working_directory=tmpdir,
                    user_dict={}
                )

class TestExecutorIntegration:
    """Tests for executor-based workflow with metadata and LAMMPS library"""

    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_executor_workflow_with_metadata_and_lammps(
        self, mock_build_config, mock_gather, mock_run_calphy
    ):
        """Test executor-based workflow with metadata and LAMMPS library"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy
        import executorlib
        import pylammpsmpi
        SingleNodeExecutor = executorlib.SingleNodeExecutor
        LammpsLibrary = pylammpsmpi.LammpsLibrary

        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})

        # Setup executor and LAMMPS library (mocked)
        mock_executor = Mock(spec=SingleNodeExecutor)
        mock_lmp = Mock(spec=LammpsLibrary)

        # Setup inputs with metadata (same as notebook executor example)
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        metadata_dict = {
            'project': 'test',
            'material': 'Al',
            'temperature': 300,
            'method': 'executor'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call with executor, LAMMPS library, and metadata
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                lmp=mock_lmp,
                metadata_dict=metadata_dict,
                user_dict={}
            )

            # Verify results
            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty

    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_metadata_validation(self, mock_build_config, mock_gather, mock_run_calphy):
        """Test that metadata dictionary is handled correctly"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})

        # Test with empty metadata (should work)
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with empty metadata
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                metadata_dict={},  # Empty metadata should be allowed
                user_dict={}
            )

            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)

    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_lammps_library_integration(self, mock_build_config, mock_gather, mock_run_calphy):
        """Test that LAMMPS library parameter is handled correctly"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy
        import pylammpsmpi
        LammpsLibrary = pylammpsmpi.LammpsLibrary

        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})

        # Setup LAMMPS library mock
        mock_lmp = Mock(spec=LammpsLibrary)

        # Test with LAMMPS library
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with LAMMPS library
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                lmp=mock_lmp,
                user_dict={}
            )

            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)

    def test_metadata_content_validation(self):
        """Test metadata content validation"""
        from phase_diagram_workflows.free_energies.ti_helpers import _validate_metadata

        # Test valid metadata
        valid_metadata = {
            'project': 'test',
            'material': 'Al',
            'temperature': 300,
            'method': 'executor'
        }
        _validate_metadata(valid_metadata)  # Should not raise

        # Test empty metadata (should be allowed)
        _validate_metadata({})  # Should not raise

        # Test metadata with extra fields (should be allowed)
        extended_metadata = {
            'project': 'test',
            'material': 'Al',
            'temperature': 300,
            'method': 'executor',
            'additional_field': 'extra_value'
        }
        _validate_metadata(extended_metadata)  # Should not raise
