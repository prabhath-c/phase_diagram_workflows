"""
Unit tests for calphy workflows.

Tests validation functions and main workflow with mocked external dependencies.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import yaml
from ase.atoms import Atoms
from ase.build import bulk

from phase_diagram_workflows.free_energies.ti_helpers import (
    _validate_input_structure,
    _validate_potential_df,
    _validate_calphy_parameters,
    _working_directory_context,
)
from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed


def _write_input_file_yaml(wd, mode="ts", reference_phase="solid", pressure=0.0, temperature=(300.0, 400.0)):
    inp = {"calculations": [{
        "mode": mode,
        "reference_phase": reference_phase,
        "pressure": pressure,
        "temperature": list(temperature),
    }]}
    with open(os.path.join(wd, "input_file.yaml"), "w") as f:
        yaml.safe_dump(inp, f)


def _write_report_yaml(wd, elements="Al", concentrations="1.0", free_energy=-3.5):
    rep = {
        "input": {"element": elements, "concentration": concentrations},
        "results": {"free_energy": free_energy},
    }
    with open(os.path.join(wd, "report.yaml"), "w") as f:
        yaml.safe_dump(rep, f)


def _write_temperature_sweep(wd, n=5):
    t = np.linspace(300.0, 400.0, n)
    fe = np.linspace(-3.5, -3.4, n)
    ferr = np.zeros(n)
    np.savetxt(os.path.join(wd, "temperature_sweep.dat"), np.column_stack([t, fe, ferr]))


def _write_ts_pair(wd, index=1, n=5):
    dx = np.linspace(0.1, 0.2, n)
    p = np.zeros(n)
    vol = np.ones(n)
    lam = np.linspace(1.0, 0.5, n)
    np.savetxt(os.path.join(wd, f"ts.forward_{index}.dat"), np.column_stack([dx, p, vol, lam]))
    np.savetxt(os.path.join(wd, f"ts.backward_{index}.dat"), np.column_stack([dx, p, vol, lam]))


class TestGatherCalphyResultsDetailed:
    """Tests for gather_calphy_results_detailed() edge cases in its own parsing logic"""

    def test_missing_input_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="input_file.yaml"):
                gather_calphy_results_detailed(tmpdir)

    def test_missing_report_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_input_file_yaml(tmpdir)
            df = gather_calphy_results_detailed(tmpdir)
            row = df.iloc[0]
            assert bool(row["status"]) is False
            assert row["composition"] is None

    def test_missing_temperature_sweep_falls_back_to_nan_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_input_file_yaml(tmpdir, temperature=(300.0, 400.0))
            _write_report_yaml(tmpdir)
            df = gather_calphy_results_detailed(tmpdir)
            row = df.iloc[0]
            assert len(row["free_energy"]) == len(row["temperature"])
            assert len(row["free_energy_error"]) == len(row["temperature"])
            assert np.all(np.isnan(row["free_energy"]))
            assert np.all(np.isnan(row["free_energy_error"]))

    def test_partial_ts_pair_missing_backward_stops_gathering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_input_file_yaml(tmpdir)
            _write_report_yaml(tmpdir)
            _write_temperature_sweep(tmpdir)
            # Only the forward file for iteration 1 exists, backward is missing.
            dx = np.linspace(0.1, 0.2, 5)
            np.savetxt(
                os.path.join(tmpdir, "ts.forward_1.dat"),
                np.column_stack([dx, np.zeros(5), np.ones(5), np.linspace(1.0, 0.5, 5)]),
            )
            df = gather_calphy_results_detailed(tmpdir)
            row = df.iloc[0]
            assert row["forward_energy_diff"] is None
            assert row["backward_energy_diff"] is None

    def test_ts_forward_backward_pairs_are_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_input_file_yaml(tmpdir)
            _write_report_yaml(tmpdir)
            _write_temperature_sweep(tmpdir)
            _write_ts_pair(tmpdir, index=1)
            df = gather_calphy_results_detailed(tmpdir)
            row = df.iloc[0]
            assert len(row["forward_energy_diff"]) == 1
            assert len(row["backward_energy_diff"]) == 1

    def test_fe_mode_missing_report_gives_nan_error_not_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_input_file_yaml(tmpdir, mode="fe", temperature=(300.0,))
            df = gather_calphy_results_detailed(tmpdir)
            row = df.iloc[0]
            assert np.isnan(row["free_energy"])
            assert np.isnan(row["free_energy_error"])

    def test_fe_mode_with_report_gives_zero_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_input_file_yaml(tmpdir, mode="fe", temperature=(300.0,))
            _write_report_yaml(tmpdir, free_energy=-3.45)
            df = gather_calphy_results_detailed(tmpdir)
            row = df.iloc[0]
            assert row["free_energy"] == -3.45
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


class TestRunCalphyDispatch:
    """Unit tests for _run_calphy's job-class dispatch logic.

    mode="composition_scaling" must always route to calphy's Alchemy class (it owns the
    switching + MC-swap + mass_integration machinery, per calphy/queuekernel.py's own
    dispatch), regardless of reference_phase -- not to Solid/Liquid, which never had that
    machinery and silently produced physically-wrong results for this mode.
    """

    @staticmethod
    def _make_input_class(mode, reference_phase, lattice_path):
        mock_input = Mock()
        mock_input.mode = mode
        mock_input.reference_phase = reference_phase
        mock_input.lattice = lattice_path
        return mock_input

    @patch('calphy.routines.routine_composition_scaling')
    @patch('calphy.Alchemy')
    @patch('calphy.Solid')
    @patch('calphy.Liquid')
    def test_composition_scaling_dispatches_to_alchemy_not_solid(
        self, mock_liquid, mock_solid, mock_alchemy, mock_routine_cs
    ):
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('composition_scaling', 'solid', lattice_path)
            mock_job = Mock()
            mock_alchemy.return_value = mock_job

            _run_calphy(input_class=input_class)

            mock_alchemy.assert_called_once_with(calculation=input_class, simfolder=tmpdir)
            mock_solid.assert_not_called()
            mock_liquid.assert_not_called()
            mock_routine_cs.assert_called_once_with(mock_job)

    @patch('calphy.routines.routine_composition_scaling')
    @patch('calphy.Alchemy')
    @patch('calphy.Liquid')
    def test_composition_scaling_dispatches_to_alchemy_not_liquid(
        self, mock_liquid, mock_alchemy, mock_routine_cs
    ):
        """reference_phase='liquid' should not matter either -- mode alone decides."""
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('composition_scaling', 'liquid', lattice_path)
            mock_alchemy.return_value = Mock()

            _run_calphy(input_class=input_class)

            mock_alchemy.assert_called_once_with(calculation=input_class, simfolder=tmpdir)
            mock_liquid.assert_not_called()

    @patch('calphy.routines.routine_composition_scaling')
    @patch('calphy.Alchemy')
    def test_composition_scaling_passes_lmp_through_to_alchemy(
        self, mock_alchemy, mock_routine_cs
    ):
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('composition_scaling', 'solid', lattice_path)
            sentinel_lmp = Mock(name='lmp')
            mock_alchemy.return_value = Mock()

            _run_calphy(input_class=input_class, lmp=sentinel_lmp)

            mock_alchemy.assert_called_once_with(
                calculation=input_class, simfolder=tmpdir, lmp=sentinel_lmp
            )

    @patch('calphy.routines.routine_fe')
    @patch('calphy.Solid')
    def test_fe_mode_still_dispatches_to_solid(self, mock_solid, mock_routine_fe):
        """Regression check: composition_scaling's new branch must not affect fe/ts."""
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('fe', 'solid', lattice_path)
            mock_job = Mock()
            mock_solid.return_value = mock_job

            _run_calphy(input_class=input_class)

            mock_solid.assert_called_once_with(calculation=input_class, simfolder=tmpdir)
            mock_routine_fe.assert_called_once_with(mock_job)
