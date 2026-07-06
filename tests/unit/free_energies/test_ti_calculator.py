"""
Unit tests for phase_diagram_workflows.free_energies.ti_calculator.

All external/heavy dependencies (_run_calphy, gather_calphy_results_detailed,
_build_calphy_config) are mocked here -- these tests check orchestration and
input-validation behavior, not real calphy execution. See
tests/integration/free_energies/test_ti_calculator.py for real-execution
coverage of gather_calphy_results_detailed.
"""

import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from ase.atoms import Atoms
from ase.build import bulk

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
